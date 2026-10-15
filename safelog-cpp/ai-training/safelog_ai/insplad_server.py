"""Explicit cropped-upper-shackle research route, preserving all old routes."""
import argparse
import binascii
import hashlib
import json
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from fastapi import HTTPException
from pydantic import BaseModel,Field,model_validator
from .real_mask_server import RealMaskEngine,create_app as parent_app
from .industrial_batch_server import Context as OldContext
from .insplad_classifier import InspladMain
from .native_component_views import normalized
from .research_delta import restore_delta
from .unified_main_server import decode
from .equipment_reference_server import hashed

CATEGORY='insplad_upper_shackle_rust'
VERSION='safelog-insplad-corrosion-v45-research'


class InspladEngine(RealMaskEngine):
    def __init__(self,parent,parent_sha,delta,delta_sha,calibration,evaluation):
        if hashlib.sha256(Path(delta).read_bytes()).hexdigest()!=delta_sha:raise ValueError('Delta SHA mismatch')
        cal=json.loads(Path(calibration).read_text(encoding='utf-8'))
        if cal['delta_sha256']!=delta_sha:raise ValueError('Calibration belongs to another delta')
        # This initializes existing policies using the original unmodified checkpoint.
        super().__init__(parent,parent_sha,evaluation)
        original=torch.load(parent,map_location='cpu',weights_only=True)
        payload=torch.load(delta,map_location='cpu',weights_only=True)
        state=restore_delta(original,payload,parent_sha)
        if payload['allowed_prefixes']!=['insplad_classifier.']:raise ValueError('Unexpected overlay namespace')
        replacement=InspladMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
        replacement.load_state_dict(state['state_dict'],strict=True)
        self.model=replacement.eval().requires_grad_(False)
        self.parent_digest=parent_sha;self.delta_digest=delta_sha
        self.digest=payload['combined_state_sha256']
        self.insplad_cut=float(cal['cut'])
        self.sources=[*self.sources,'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1, cropped upper-shackle rust labels, CC BY-NC3.0 noncommercial academic research.']
        self.insplad_policy=hashed({'version':VERSION,'category':CATEGORY,'parentCheckpointSha256':parent_sha,'deltaSha256':delta_sha,
            'modelIdentityType':'combined_tensor_state_sha256','sourceScope':'cropped_polymer_insulator_upper_metal_shackle_visible_rust_only',
            'input':'RGBbicubic256square_ImageNet','descriptor':'same frozen fine24_low40_high960 mean_popstd_max3072 FP16roundtrip',
            'threshold':self.insplad_cut,'normalCAL':cal['normal_photos'],'normalCALRank':cal['rank'],'normalCALAlpha':.15,
            'actualOptimizerUpdates':1920,'selectedEpoch':4,'sharedBackboneCount':1,'neuralModelCount':1,
            'scoreIsCalibratedProbability':False,'localizationAvailable':False,'inputDomainAutoVerified':False,
            'captureDateLabelCorrelation':True,'physicalPartIndependenceVerified':False,'independentFieldValidation':False})

    def analyze(self,task,encoded,category=None):
        if task==CATEGORY or (task=='equipment_surface' and category==CATEGORY):
            rgb,raw=decode(encoded)
            if min(rgb.size)<64:raise ValueError('Upper-shackle crop at least64 pixels required; review input')
            with self.lock,torch.inference_mode():
                x=normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None]
                score=float(self.model.forward_insplad_upper_shackle(x).sigmoid()[0])
            result={'schema':'safelog.insplad-corrosion-photo.v1','inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
                'classResults':[{'className':CATEGORY,'score':score,'threshold':self.insplad_cut,'candidateDetected':score>self.insplad_cut}],
                'candidateDetected':score>self.insplad_cut,'inputSha256':hashlib.sha256(raw).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
                'requiredInput':'cropped_polymer_insulator_upper_metal_shackle','inputDomainAutoVerified':False,
                'scoreIsCalibratedProbability':False,'localizationAvailable':False,'heatmapIsConfirmedDefectLocalization':False,
                'mechanicalFailureState':'not_assessed','electricalHealthState':'not_assessed',
                'inferencePolicyVersion':VERSION,'inferencePolicySha256':self.insplad_policy['sha256'],'backbonePassesPerPhoto':1,
                'sourceBenchmark':{'normal':31,'rust':33,'fp':6,'fn':0,'fpr':6/31,'fnr':0.0},
                'sourceBenchmarkScope':'reserved_publisherVAL_crops_with_frame_near_group_exclusion_not_independent_field',
                'captureDateLabelCorrelation':True,'physicalPartIndependenceVerified':False,
                'humanConfirmationRequired':True,'fieldPerformanceValidated':False,'independentFieldValidation':False,
                'researchPrototype':True,'otherTasksEvaluated':False,'neuralModelCount':1,'sharedBackboneCount':1,
                'modelName':VERSION,'modelSha256':self.digest,'trainedDatasetSources':self.sources}
        else:result=super().analyze(task,encoded,category)
        result.update(parentCheckpointSha256=self.parent_digest,deltaSha256=self.delta_digest,modelIdentityType='combined_tensor_state_sha256')
        return result


class Context(BaseModel):
    inspectionTask:str=Field(min_length=1,max_length=80)
    equipmentCategory:Optional[str]=Field(default=None,max_length=80)
    promptVersion:str=Field(min_length=1,max_length=80)
    @model_validator(mode='after')
    def valid_task(self):
        if self.inspectionTask==CATEGORY:
            if self.equipmentCategory not in (None,CATEGORY):raise ValueError('Alias/category must agree')
        elif self.inspectionTask=='equipment_surface' and self.equipmentCategory==CATEGORY:pass
        else:OldContext(**self.model_dump())
        return self


class Hazard(Context):
    image:str=Field(min_length=1,max_length=12000000)
    userMemo:str=Field(default='',max_length=10000)


class Compare(Context):
    beforeImage:str=Field(min_length=1,max_length=12000000)
    afterImage:str=Field(min_length=1,max_length=12000000)
    actionNote:str=Field(default='',max_length=10000)


def create_app(engine):
    app=parent_app(engine)
    paths=('/v1/analyze-hazard','/v1/main/analyze','/v1/compare-action','/v1/equipment-categories','/v1/inference-policy')
    old={p:next(r.endpoint for r in app.router.routes if getattr(r,'path',None)==p) for p in paths}
    app.router.routes=[r for r in app.router.routes if getattr(r,'path',None) not in paths]
    def is_new(body):return body.inspectionTask==CATEGORY or body.equipmentCategory==CATEGORY
    def analyze(body,image):
        try:return engine.analyze(body.inspectionTask,image,body.equipmentCategory)
        except (ValueError,binascii.Error) as error:raise HTTPException(422,str(error)) from error
    @app.post('/v1/analyze-hazard')
    @app.post('/v1/main/analyze')
    def hazard(body:Hazard):
        if not is_new(body):return old['/v1/analyze-hazard'](body)
        value=analyze(body,body.image);names=['상부 금속 연결고리 부식 후보'] if value['candidateDetected'] else []
        value.update(hazardCategory=names[0] if names else '외관 이상 후보 미검출',riskLevel=3 if names else 1,detectedHazards=names,
            suggestedDescription='상부 금속 연결고리의 사진상 부식 후보를 점검자가 확인하세요.' if names else '해당 사진에서 부식 후보를 찾지 못했습니다.',
            suggestedAction='현장 점검자가 부품의 상태를 확인하고 필요한 조치와 재점검 내용을 기록하세요.',
            confidence=value['classResults'][0]['score'],confidenceMethod='sigmoid_score_not_calibrated_probability',
            riskLevelMethod='rule_template_not_learned_severity',promptVersion=body.promptVersion)
        return value
    @app.post('/v1/compare-action')
    def compare(body:Compare):
        if not is_new(body):return old['/v1/compare-action'](body)
        before=analyze(body,body.beforeImage);after=analyze(body,body.afterImage)
        return {'likelyResolved':before['candidateDetected'] and not after['candidateDetected'],
            'remainingRisks':['부식 후보'] if after['candidateDetected'] else [],'assessment':'동일한 기준의 사진상 후보 변화이며 점검자가 최종 확인해야 합니다.',
            'confidence':.5,'modelName':VERSION,'modelSha256':engine.digest,'inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
            'beforeCandidateDetected':before['candidateDetected'],'afterCandidateDetected':after['candidateDetected'],'humanConfirmationRequired':True,
            'mechanicalFailureState':'not_assessed','scoreIsCalibratedProbability':False,'comparisonMethod':'same_fixed_model_scores_not_trained_change_detector',
            'promptVersion':body.promptVersion,'parentCheckpointSha256':engine.parent_digest,'deltaSha256':engine.delta_digest}
    @app.get('/v1/equipment-categories')
    def categories():
        value=old['/v1/equipment-categories']()
        value['categories'].append({'id':CATEGORY,'label':'고분자 애자 상부 금속 연결고리 부식','trained':True,
            'researchPrototype':True,'requiredInput':'cropped_polymer_insulator_upper_metal_shackle','inputDomainAutoVerified':False,
            'localizationAvailable':False,'sourceBenchmarkScope':'reserved_publisherVAL_only','independentFieldValidation':False,
            'benchmarkStage':{'state':'source_photo_20percent_only','ready_for_next_stage':True}})
        return value
    @app.get('/v1/inference-policy')
    def policies():
        value=old['/v1/inference-policy']();value['inspladCorrosion']={CATEGORY:engine.insplad_policy};return value
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','parent-sha','delta','delta-sha','calibration','evaluation'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--port',type=int,default=8775);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import uvicorn
    uvicorn.run(create_app(InspladEngine(args.parent,args.parent_sha,args.delta,args.delta_sha,args.calibration,args.evaluation)),host='127.0.0.1',port=args.port)
