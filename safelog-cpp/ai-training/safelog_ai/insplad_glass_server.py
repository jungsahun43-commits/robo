"""Explicit glass-source route; prior component tasks retain their policies/maps."""
import argparse
import binascii
import hashlib
import json
from pathlib import Path
from typing import Optional
import torch
from fastapi import HTTPException
from pydantic import BaseModel,Field,model_validator
from .insplad_server import InspladEngine,Context as PriorContext,create_app as prior_app
from .insplad_glass import InspladGlassMain,glass_view,assert_preserved
from .native_component_views import normalized
from .research_delta import restore_delta
from .unified_main_server import decode
from .equipment_reference_server import hashed

CATEGORY='insplad_glass_missing_cap'
VERSION='safelog-insplad-glass-v48-research'
EXPECTED_DELTA='520b010585708f2bbf008ab6354a8dde53a64f3a2db9721b6475ea599022dafb'


class GlassEngine(InspladEngine):
    def __init__(self,parent,parent_sha,corrosion_delta,corrosion_delta_sha,corrosion_calibration,delta,delta_sha,calibration,evaluation):
        if delta_sha!=EXPECTED_DELTA or hashlib.sha256(Path(delta).read_bytes()).hexdigest()!=delta_sha:
            raise ValueError('Registered glass research delta SHA required')
        cal=json.loads(Path(calibration).read_text(encoding='utf-8'))
        if cal['delta_sha256']!=delta_sha or cal['rank']!=74 or cal['normal_photos']!=85:
            raise ValueError('Fixed glass calibration required')
        super().__init__(parent,parent_sha,corrosion_delta,corrosion_delta_sha,corrosion_calibration,evaluation)
        original=torch.load(parent,map_location='cpu',weights_only=True)
        approved=restore_delta(original,torch.load(corrosion_delta,map_location='cpu',weights_only=True),parent_sha)
        payload=torch.load(delta,map_location='cpu',weights_only=True)
        if payload['allowed_prefixes']!=['insplad_classifier.','insplad_glass_classifier.']:raise ValueError('Unexpected composite namespaces')
        state=restore_delta(original,payload,parent_sha);assert_preserved(approved,state)
        replacement=InspladGlassMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
        replacement.load_state_dict(state['state_dict'],strict=True)
        self.model=replacement.eval().requires_grad_(False)
        self.digest=payload['combined_state_sha256'];self.delta_digest=delta_sha;self.glass_cut=float(cal['cut'])
        self.glass_policy=hashed({'version':VERSION,'category':CATEGORY,'parentCheckpointSha256':parent_sha,'deltaSha256':delta_sha,
            'preservedCorrosionDeltaSha256':corrosion_delta_sha,'modelIdentityType':'combined_tensor_state_sha256',
            'requiredInput':'cropped_glass_insulator_chain','input':'isotropicRGB384letterbox127_ImageNet',
            'descriptor':'shared fine24/low40/high960 FP16roundtrip, pool cellcenters within unpadded input rectangle, mean/popstd/max3072 FP16roundtrip',
            'inputRectangleIsDefectOrObjectGT':False,'convolutionReceptiveFieldsCanIncludePadding':True,
            'threshold':self.glass_cut,'normalCAL':85,'normalCALRank':74,'normalCALAlpha':.15,
            'actualOptimizerUpdates':1920,'selectedEpoch':1,'sharedBackboneCount':1,'neuralModelCount':1,
            'sourcePhotoFP':2,'sourcePhotoFN':6,'sourceNormalPhotos':29,'sourceDefectPhotos':30,
            'scoreIsCalibratedProbability':False,'localizationAvailable':False,'inputDomainAutoVerified':False,
            'physicalPartIndependenceVerified':False,'independentFieldValidation':False})

    def analyze(self,task,encoded,category=None):
        if task!=CATEGORY and not (task=='equipment_surface' and category==CATEGORY):return super().analyze(task,encoded,category)
        rgb,raw=decode(encoded)
        if min(rgb.size)<64:raise ValueError('Glass-insulator crop at least64pixels required; review input')
        view,box=glass_view(rgb)
        with self.lock,torch.inference_mode():score=float(self.model.forward_insplad_glass(normalized(view)[None],[box]).sigmoid()[0])
        return {'schema':'safelog.insplad-glass-photo.v1','inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
            'classResults':[{'className':CATEGORY,'score':score,'threshold':self.glass_cut,'candidateDetected':score>self.glass_cut}],
            'candidateDetected':score>self.glass_cut,'inputSha256':hashlib.sha256(raw).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
            'requiredInput':'cropped_glass_insulator_chain','inputDomainAutoVerified':False,'scoreIsCalibratedProbability':False,
            'localizationAvailable':False,'heatmapIsConfirmedDefectLocalization':False,'mechanicalFailureState':'not_assessed','electricalHealthState':'not_assessed',
            'inferencePolicyVersion':VERSION,'inferencePolicySha256':self.glass_policy['sha256'],'backbonePassesPerPhoto':1,
            'nativeViewGeometry':{'side':384,'unpaddedRectangle':box,'rectangleIsDefectGT':False},
            'sourceBenchmark':{'normal':29,'missingCap':30,'fp':2,'fn':6,'fpr':2/29,'fnr':6/30},
            'sourceBenchmarkScope':'reserved_publisherVAL_crops_not_independent_field','physicalPartIndependenceVerified':False,
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
        result=analyze(body,body.image);names=['유리 애자 캡 누락 후보'] if result['candidateDetected'] else []
        result.update(hazardCategory=names[0] if names else '외관 이상 후보 미검출',riskLevel=3 if names else 1,detectedHazards=names,
            suggestedDescription='사진에서 유리 애자의 누락 후보를 점검자가 확인하세요.' if names else '해당 사진에서 누락 후보를 찾지 못했습니다.',
            suggestedAction='현장 점검자가 부품 상태를 확인하고 필요한 조치와 재점검 내용을 기록하세요.',
            confidence=result['classResults'][0]['score'],confidenceMethod='sigmoid_score_not_calibrated_probability',
            riskLevelMethod='rule_template_not_learned_severity',promptVersion=body.promptVersion)
        return result
    @app.post('/v1/compare-action')
    def compare(body:Compare):
        if not is_new(body):return old['/v1/compare-action'](body)
        before=analyze(body,body.beforeImage);after=analyze(body,body.afterImage)
        return {'likelyResolved':before['candidateDetected'] and not after['candidateDetected'],
            'remainingRisks':['유리 애자 누락 후보'] if after['candidateDetected'] else [],'assessment':'같은 기준의 사진상 후보 변화이며 현장 점검자가 최종 확인해야 합니다.',
            'confidence':.5,'modelName':VERSION,'modelSha256':engine.digest,'inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,
            'beforeCandidateDetected':before['candidateDetected'],'afterCandidateDetected':after['candidateDetected'],
            'humanConfirmationRequired':True,'mechanicalFailureState':'not_assessed','electricalHealthState':'not_assessed',
            'scoreIsCalibratedProbability':False,'comparisonMethod':'same_fixed_model_scores_not_trained_change_detector','promptVersion':body.promptVersion}
    @app.get('/v1/equipment-categories')
    def categories():
        value=old['/v1/equipment-categories']();value['categories'].append({'id':CATEGORY,'label':'유리 애자 캡 누락','trained':True,
            'researchPrototype':True,'requiredInput':'cropped_glass_insulator_chain','inputDomainAutoVerified':False,'localizationAvailable':False,
            'sourceBenchmarkScope':'reserved_publisherVAL_only','independentFieldValidation':False,
            'benchmarkStage':{'state':'source_photo_20percent_only','ready_for_next_stage':True}});return value
    @app.get('/v1/inference-policy')
    def policies():
        value=old['/v1/inference-policy']();value['inspladGlass']={CATEGORY:engine.glass_policy};return value
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','parent-sha','corrosion-delta','corrosion-delta-sha','corrosion-calibration','delta','delta-sha','calibration','evaluation'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--port',type=int,default=8775);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import uvicorn
    uvicorn.run(create_app(GlassEngine(args.parent,args.parent_sha,args.corrosion_delta,args.corrosion_delta_sha,args.corrosion_calibration,args.delta,args.delta_sha,args.calibration,args.evaluation)),host='127.0.0.1',port=args.port)
