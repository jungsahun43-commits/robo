"""Explicit suspension-crop rust route, preserving both approved source tasks."""
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
from .insplad_glass_server import GlassEngine,Context as PriorContext,create_app as prior_app
from .insplad_glass import assert_preserved
from .insplad_suspension import InspladSuspensionMain
from .native_component_views import normalized
from .research_delta import restore_delta
from .unified_main_server import decode
from .equipment_reference_server import hashed

CATEGORY='insplad_lightning_suspension_rust'
VERSION='safelog-insplad-suspension-v51-research'
EXPECTED_DELTA='b91b6578eccd1886c5be18805a9475875ff319f77bda019cd0b3f518d84f46dc'
EXPECTED_CAL='0edaac4b637c0ffb7252fbdd7fd0a023fff368286bec3d21769b7dda5bd2f590'


class SuspensionEngine(GlassEngine):
    def __init__(self,parent,parent_sha,corrosion_delta,corrosion_delta_sha,corrosion_calibration,
                 glass_delta,glass_delta_sha,glass_calibration,delta,delta_sha,calibration,evaluation):
        if delta_sha!=EXPECTED_DELTA or hashlib.sha256(Path(delta).read_bytes()).hexdigest()!=delta_sha:
            raise ValueError('Registered suspension research delta SHA required')
        if hashlib.sha256(Path(calibration).read_bytes()).hexdigest()!=EXPECTED_CAL:
            raise ValueError('Registered suspension calibration SHA required')
        cal=json.loads(Path(calibration).read_text(encoding='utf-8'))
        if cal['delta_sha256']!=delta_sha or cal['rank']!=45 or cal['normal_photos']!=51:
            raise ValueError('Fixed suspension calibration required')
        super().__init__(parent,parent_sha,corrosion_delta,corrosion_delta_sha,corrosion_calibration,
                         glass_delta,glass_delta_sha,glass_calibration,evaluation)
        original=torch.load(parent,map_location='cpu',weights_only=True)
        approved=restore_delta(original,torch.load(glass_delta,map_location='cpu',weights_only=True),parent_sha)
        payload=torch.load(delta,map_location='cpu',weights_only=True)
        if payload['allowed_prefixes']!=['insplad_classifier.','insplad_glass_classifier.','insplad_suspension_classifier.']:
            raise ValueError('Unexpected composite namespaces')
        state=restore_delta(original,payload,parent_sha);assert_preserved(approved,state)
        replacement=InspladSuspensionMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
        replacement.load_state_dict(state['state_dict'],strict=True)
        self.model=replacement.eval().requires_grad_(False)
        self.digest=payload['combined_state_sha256'];self.delta_digest=delta_sha;self.suspension_cut=float(cal['cut'])
        self.suspension_policy=hashed({'version':VERSION,'category':CATEGORY,'parentCheckpointSha256':parent_sha,'deltaSha256':delta_sha,
            'preservedGlassDeltaSha256':glass_delta_sha,'preservedCorrosionDeltaSha256':corrosion_delta_sha,
            'modelIdentityType':'combined_tensor_state_sha256','requiredInput':'cropped_lightning_rod_suspension',
            'input':'RGBbicubic256square_ImageNet','descriptor':'same frozen fine24_low40_high960 mean_popstd_max3072 FP16roundtrip',
            'threshold':self.suspension_cut,'normalCAL':51,'normalCALRank':45,'normalCALAlpha':.15,
            'actualOptimizerUpdates':1920,'selectedEpoch':2,'sharedBackboneCount':1,'neuralModelCount':1,
            'sourcePhotoFP':24,'sourcePhotoFN':0,'sourceNormalPhotos':231,'sourceDefectPhotos':20,
            'scoreIsCalibratedProbability':False,'localizationAvailable':False,'inputDomainAutoVerified':False,
            'captureDateLabelCorrelation':True,'evaluationDatesWithBothLabels':0,
            'physicalPartIndependenceVerified':False,'independentFieldValidation':False})

    def analyze(self,task,encoded,category=None):
        if task!=CATEGORY and not (task=='equipment_surface' and category==CATEGORY):return super().analyze(task,encoded,category)
        rgb,raw=decode(encoded)
        if min(rgb.size)<64:raise ValueError('Suspension crop at least64pixels required; review input')
        with self.lock,torch.inference_mode():
            x=normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None]
            score=float(self.model.forward_insplad_suspension(x).sigmoid()[0])
        return {'schema':'safelog.insplad-suspension-photo.v1','inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
            'classResults':[{'className':CATEGORY,'score':score,'threshold':self.suspension_cut,'candidateDetected':score>self.suspension_cut}],
            'candidateDetected':score>self.suspension_cut,'inputSha256':hashlib.sha256(raw).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
            'requiredInput':'cropped_lightning_rod_suspension','inputDomainAutoVerified':False,'scoreIsCalibratedProbability':False,
            'localizationAvailable':False,'heatmapIsConfirmedDefectLocalization':False,'mechanicalFailureState':'not_assessed','electricalHealthState':'not_assessed',
            'inferencePolicyVersion':VERSION,'inferencePolicySha256':self.suspension_policy['sha256'],'backbonePassesPerPhoto':1,
            'sourceBenchmark':{'normal':231,'rust':20,'fp':24,'fn':0,'fpr':24/231,'fnr':0.0},
            'sourceBenchmarkScope':'reserved_publisherVAL_crops_not_independent_field','physicalPartIndependenceVerified':False,
            'captureDateLabelCorrelation':True,'evaluationDatesWithBothLabels':0,
            'humanConfirmationRequired':True,'fieldPerformanceValidated':False,'independentFieldValidation':False,'researchPrototype':True,
            'otherTasksEvaluated':False,'neuralModelCount':1,'sharedBackboneCount':1,'modelName':VERSION,'modelSha256':self.digest,
            'parentCheckpointSha256':self.parent_digest,'deltaSha256':self.delta_digest,'modelIdentityType':'combined_tensor_state_sha256'}


class Context(BaseModel):
    inspectionTask:str=Field(min_length=1,max_length=80)
    equipmentCategory:Optional[str]=Field(default=None,max_length=80)
    promptVersion:str=Field(min_length=1,max_length=80)
    @model_validator(mode='after')
    def valid_task(self):
        if self.inspectionTask==CATEGORY:
            if self.equipmentCategory not in (None,CATEGORY):raise ValueError('Alias/category must agree')
        elif self.inspectionTask=='equipment_surface' and self.equipmentCategory==CATEGORY:pass
        else:PriorContext(**self.model_dump())
        return self


class Hazard(Context):
    image:str=Field(min_length=1,max_length=12000000)
    userMemo:str=Field(default='',max_length=10000)


class Compare(Context):
    beforeImage:str=Field(min_length=1,max_length=12000000)
    afterImage:str=Field(min_length=1,max_length=12000000)
    actionNote:str=Field(default='',max_length=10000)


def create_app(engine):
    app=prior_app(engine)
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
        result=analyze(body,body.image);names=['현수 연결구 부식 후보'] if result['candidateDetected'] else []
        result.update(hazardCategory=names[0] if names else '외관 이상 후보 미검출',riskLevel=3 if names else 1,detectedHazards=names,
            suggestedDescription='사진에서 현수 연결구의 부식 후보를 점검자가 확인하세요.' if names else '해당 사진에서 부식 후보를 찾지 못했습니다.',
            suggestedAction='현장 점검자가 부품 상태를 확인하고 필요한 조치와 재점검 내용을 기록하세요.',
            confidence=result['classResults'][0]['score'],confidenceMethod='sigmoid_score_not_calibrated_probability',
            riskLevelMethod='rule_template_not_learned_severity',promptVersion=body.promptVersion)
        return result
    @app.post('/v1/compare-action')
    def compare(body:Compare):
        if not is_new(body):return old['/v1/compare-action'](body)
        before=analyze(body,body.beforeImage);after=analyze(body,body.afterImage)
        return {'likelyResolved':before['candidateDetected'] and not after['candidateDetected'],
            'remainingRisks':['현수 연결구 부식 후보'] if after['candidateDetected'] else [],'assessment':'같은 기준의 사진상 후보 변화이며 현장 점검자가 최종 확인해야 합니다.',
            'confidence':.5,'modelName':VERSION,'modelSha256':engine.digest,'inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
            'beforeCandidateDetected':before['candidateDetected'],'afterCandidateDetected':after['candidateDetected'],
            'humanConfirmationRequired':True,'mechanicalFailureState':'not_assessed','electricalHealthState':'not_assessed',
            'scoreIsCalibratedProbability':False,'comparisonMethod':'same_fixed_model_scores_not_trained_change_detector','promptVersion':body.promptVersion}
    @app.get('/v1/equipment-categories')
    def categories():
        value=old['/v1/equipment-categories']();value['categories'].append({'id':CATEGORY,'label':'현수 연결구 부식','trained':True,
            'researchPrototype':True,'requiredInput':'cropped_lightning_rod_suspension','inputDomainAutoVerified':False,'localizationAvailable':False,
            'sourceBenchmarkScope':'reserved_publisherVAL_only','independentFieldValidation':False,
            'benchmarkStage':{'state':'source_photo_20percent_only','ready_for_next_stage':True}});return value
    @app.get('/v1/inference-policy')
    def policies():
        value=old['/v1/inference-policy']();value['inspladSuspension']={CATEGORY:engine.suspension_policy};return value
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    names=('parent','parent-sha','corrosion-delta','corrosion-delta-sha','corrosion-calibration','glass-delta','glass-delta-sha','glass-calibration','delta','delta-sha','calibration','evaluation')
    for name in names:parser.add_argument('--'+name,required=True)
    parser.add_argument('--port',type=int,default=8775);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import uvicorn
    uvicorn.run(create_app(SuspensionEngine(**{name.replace('-','_'):getattr(args,name.replace('-','_')) for name in names})),host='127.0.0.1',port=args.port)
