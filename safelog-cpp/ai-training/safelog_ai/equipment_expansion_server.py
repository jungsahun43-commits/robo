"""Existing app operations plus explicit equipment categories, ONE checkpoint."""
import argparse
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
from typing import Optional
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from .equipment_expansion import ARCH, CATEGORIES, SIZE, ExpandedMain, photo_score
from .unified_main import TASKS
from .unified_main_server import decode, LABELS as OLD_LABELS
from .unified_tiles_server import TilesEngine
from .belt_tiles import policy

LABELS = {**OLD_LABELS, **{c: label for c, label in zip(CATEGORIES,
    ('케이블 외관 이상','나사 외관 이상','너트 외관 이상','전선 연결부 외관 이상',
     '검정 브래킷 외관 이상','갈색 브래킷 외관 이상','흰색 브래킷 외관 이상',
     '금속 연결부 외관 이상','금속판 외관 이상','금속 튜브 외관 이상','부품 표면 이상'))}}


def prepare(rgb):
    values = torch.from_numpy(np.array(rgb.resize((SIZE, SIZE), Image.Resampling.BICUBIC))).permute(2,0,1).float()[None]/255
    return (values-values.new_tensor([.485,.456,.406])[None,:,None,None])/values.new_tensor([.229,.224,.225])[None,:,None,None]


class ExpansionEngine(TilesEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw=Path(checkpoint).read_bytes(); self.digest=hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit prototype checkpoint SHA mismatch')
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if state.get('architecture')!=ARCH or not state.get('one_shared_backbone') or state.get('equipment_categories')!=list(CATEGORIES):
            raise ValueError('One-backbone equipment prototype checkpoint required')
        self.model=ExpandedMain().eval();self.model.load_state_dict(state['state_dict'],strict=True)
        self.lock=RLock();self.thresholds=state['thresholds'];self.points=state['equipment_operating_points']
        self.sources=state['trained_dataset_sources'];self.mode='global_tile_mean';self.inference_policy=policy(self.mode)
        self.evaluation={}
        if evaluation:
            info=json.loads(Path(evaluation).read_text(encoding='utf-8'))
            if info['checkpoint_sha256']!=self.digest:raise ValueError('Evaluation must belong to this exact checkpoint')
            self.evaluation=info['categories']
        setting={'version':'safelog-equipment-expansion-v1','preprocess':'bicubic RGB256 square, ImageNet normalization',
                 'score':'mean top1percent raw cosine residual','normal_calibration':'category-specific normalCAL85percent order statistic',
                 'oneSharedBackbone':True,'heatmapIsConfirmedDefectLocalization':False}
        self.equipment_policy={**setting,'sha256':hashlib.sha256(json.dumps(setting,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

    def analyze_equipment(self, category, encoded):
        if category not in self.points:raise ValueError('Category not trained/calibrated; cannot infer as normal')
        rgb,raw=decode(encoded)
        with self.lock,torch.inference_mode():heat=self.model.forward_equipment(prepare(rgb),category)
        score=float(photo_score(heat)[0]); threshold=self.points[category]['photo_threshold'];candidate=score>threshold
        image=Image.fromarray((heat[0].clamp(0,2).numpy()/2*255).astype('uint8')).resize(rgb.size,Image.Resampling.BILINEAR)
        buffer=io.BytesIO();image.save(buffer,format='PNG')
        return {'schema':'safelog.equipment-expansion.v1','inspectionTask':'equipment_surface','equipmentCategory':category,
            'classResults':[{'className':category,'score':score,'threshold':threshold,'candidateDetected':candidate}],
            'candidateDetected':candidate,'inputSha256':hashlib.sha256(raw).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
            'heatmapPngBase64':base64.b64encode(buffer.getvalue()).decode(),'heatmapIsConfirmedDefectLocalization':False,
            'scoreIsCalibratedProbability':False,'evidenceScope':'coarse_component_appearance_feature_residual',
            'inferencePolicyVersion':self.equipment_policy['version'],'inferencePolicySha256':self.equipment_policy['sha256'],
            'benchmarkStage':self.evaluation.get(category,{}).get('stage_gate',{'state':'not_evaluated','ready_for_next_stage':False}),
            'normalCALCount':self.points[category]['normal_CAL_photos'],'mechanicalFailureState':'not_assessed'}

    def analyze(self, task, encoded, category=None):
        if task in CATEGORIES:category,task=task,'equipment_surface'
        if task=='equipment_surface':result=self.analyze_equipment(category,encoded)
        elif task in TASKS:result=super().analyze(task,encoded)
        else:raise ValueError('Explicit supported inspection task required')
        result.update(neuralModelCount=1,sharedBackboneCount=1,modelName=ARCH,modelSha256=self.digest,
            trainedDatasetSources=self.sources,otherTasksEvaluated=False,humanConfirmationRequired=True,
            fieldPerformanceValidated=False,researchPrototype=True)
        return result


class Context(BaseModel):
    inspectionTask:str=Field(min_length=1,max_length=80)
    equipmentCategory:Optional[str]=Field(default=None,max_length=80)
    promptVersion:str=Field(min_length=1,max_length=80)
    @model_validator(mode='after')
    def valid_task(self):
        if self.inspectionTask not in (*TASKS,'equipment_surface',*CATEGORIES):raise ValueError('Supported task required')
        if self.inspectionTask=='equipment_surface' and self.equipmentCategory not in CATEGORIES:raise ValueError('Explicit equipment category required')
        if self.inspectionTask in CATEGORIES and self.equipmentCategory not in (None,self.inspectionTask):raise ValueError('Category alias and category must agree')
        if self.inspectionTask in TASKS and self.equipmentCategory is not None:raise ValueError('Legacy task does not accept equipment category')
        return self

class Hazard(Context):
    image:str=Field(min_length=1,max_length=12000000)
    userMemo:str=Field(default='',max_length=10000)
class Compare(Context):
    beforeImage:str=Field(min_length=1,max_length=12000000)
    afterImage:str=Field(min_length=1,max_length=12000000)
    actionNote:str=Field(default='',max_length=10000)
class Summary(BaseModel):
    inspectionContext:str=Field(min_length=1,max_length=20000)
    promptVersion:str=Field(min_length=1,max_length=80)


def create_expansion_app(engine=None):
    app=FastAPI(title='SafeLog equipment prototype',version='1.0.0')
    def analyze(body, image):
        if engine is None:raise HTTPException(503,'Explicit checkpoint and SHA required')
        try:return engine.analyze(body.inspectionTask,image,body.equipmentCategory)
        except (ValueError,binascii.Error) as error:raise HTTPException(422,str(error)) from error
    @app.get('/health')
    def health():return {'ready':engine is not None,'neuralModelCount':1 if engine else 0,'sharedBackboneCount':1 if engine else 0,
        'modelSha256':engine.digest if engine else None,'inspectionTasks':list(TASKS)+['equipment_surface'],'researchPrototype':True,'fieldPerformanceValidated':False}
    @app.get('/v1/equipment-categories')
    def categories():return {'categories':[{'id':c,'label':LABELS[c],'trained':bool(engine and c in engine.points),
        'benchmarkStage':engine.evaluation.get(c,{}).get('stage_gate',{'state':'not_evaluated','ready_for_next_stage':False}) if engine else {'state':'not_loaded','ready_for_next_stage':False}} for c in CATEGORIES]}
    @app.get('/v1/inference-policy')
    def policies():
        if engine is None:raise HTTPException(503,'Model not loaded')
        return {'legacyBelt':engine.inference_policy,'equipment':engine.equipment_policy}
    @app.post('/v1/analyze-hazard')
    @app.post('/v1/main/analyze')
    def hazard(body:Hazard):
        result=analyze(body,body.image);names=[LABELS[x['className']]+' 후보' for x in result['classResults'] if x['candidateDetected']]
        raw=max(x['score'] for x in result['classResults']);confidence=min(1.,max(0.,raw/2 if result['inspectionTask']=='equipment_surface' else raw))
        result.update(hazardCategory=' / '.join(names) if names else '외관 이상 후보 미검출',riskLevel=3 if names else 1,detectedHazards=names,
            suggestedDescription='사진에서 '+', '.join(names)+'가 제안되었습니다.' if names else '해당 점검 항목에서 외관 이상 후보를 찾지 못했습니다.',
            suggestedAction='점검자가 사진과 현장을 확인하고 담당자의 조치 및 재점검 내용을 기록하세요.',confidence=confidence,
            confidenceMethod='bounded_score_not_calibrated_probability',riskLevelMethod='rule_template_not_learned_severity',promptVersion=body.promptVersion)
        return result
    @app.post('/v1/compare-action')
    def compare(body:Compare):
        before=analyze(body,body.beforeImage);after=analyze(body,body.afterImage);resolved=before['candidateDetected'] and not after['candidateDetected']
        return {'likelyResolved':resolved,'remainingRisks':[LABELS[x['className']]+' 후보' for x in after['classResults'] if x['candidateDetected']],
            'assessment':'사진의 외관 후보 변화를 확인하고 점검자가 최종 조치 상태를 판단해야 합니다.','confidence':0.5,
            'modelName':ARCH,'modelSha256':engine.digest,'inspectionTask':after['inspectionTask'],'equipmentCategory':after.get('equipmentCategory'),
            'beforeCandidateDetected':before['candidateDetected'],'afterCandidateDetected':after['candidateDetected'],
            'humanConfirmationRequired':True,'mechanicalFailureState':'not_assessed','scoreIsCalibratedProbability':False,
            'comparisonMethod':'same_fixed_model_scores_not_trained_change_detector','promptVersion':body.promptVersion}
    @app.post('/v1/summarize')
    def summary(body:Summary):return {'summary':'점검·조치·재점검 기록 요약\n'+body.inspectionContext,'keyRisks':[],
        'generationMethod':'rule_template_not_trained_LLM','modelName':ARCH+'/report-template','promptVersion':body.promptVersion,'humanConfirmationRequired':True}
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--evaluation');parser.add_argument('--port',type=int,default=8775);args=parser.parse_args();torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_expansion_app(ExpansionEngine(args.checkpoint,args.sha256,args.evaluation)),host='127.0.0.1',port=args.port)
