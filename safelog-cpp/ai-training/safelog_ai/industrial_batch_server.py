"""One-checkpoint supervised wear plus preserved and extended component evidence."""
import argparse
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
from typing import Optional
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from .industrial_supervised import ARCH as WEAR_ARCH, WEAR_CATEGORY, IndustrialSupervisedMain, wear_descriptor
from .industrial_surface_reference import ARCH, IndustrialSurfaceMain
from .native_component_server import NativeComponentEngine, encode_png
from .native_component_views import normalized
from .equipment_expansion import CATEGORIES
from .equipment_expansion_server import prepare, LABELS as ORIGINAL_LABELS
from .equipment_reference_server import hashed, png
from .unified_main import TASKS
from .unified_main_server import decode
from .belt_tiles import policy

ALL_CATEGORIES = (*CATEGORIES, WEAR_CATEGORY)
LABELS = {**ORIGINAL_LABELS, WEAR_CATEGORY: '볼스크루 표면 피팅 손상'}


class IndustrialEngine(NativeComponentEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw = Path(checkpoint).read_bytes(); self.digest = hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit research checkpoint SHA mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (state.get('architecture') not in (ARCH, WEAR_ARCH) or not state.get('one_shared_backbone') or
            state.get('equipment_categories') != list(CATEGORIES) or state.get('wear_category') != WEAR_CATEGORY or
            state.get('native_screw_budget_variant') != 'normalCAL15' or not state.get('wear_protocol_sha256')):
            raise ValueError('Selected one-backbone supervised industrial checkpoint required')
        self.surface_specs = state.get('surface_reference_specs', {})
        if state['architecture'] == ARCH:
            self.model = IndustrialSurfaceMain(state['reference_counts'], state['native_specs'], state['wear_variant'], self.surface_specs)
        else:
            self.model = IndustrialSupervisedMain(state['reference_counts'], state['native_specs'], state['wear_variant'])
        self.model.eval(); self.model.load_state_dict(state['state_dict'], strict=True)
        self.lock = RLock(); self.thresholds = state['thresholds']; self.points = state['equipment_operating_points']; self.sources = state['trained_dataset_sources']
        self.mode = 'global_tile_mean'; self.inference_policy = policy(self.mode); self.evaluation = {}
        if evaluation:
            value = json.loads(Path(evaluation).read_text(encoding='utf-8'))
            if value['checkpoint_sha256'] != self.digest: raise ValueError('Evaluation must belong to exact checkpoint')
            self.evaluation = value['categories']
        self.equipment_policy = hashed({'version': 'safelog-equipment-expansion-v1', 'preprocess': 'bicubic RGB256 square, ImageNet normalization',
            'score': 'mean top1percent raw cosine residual', 'normal_calibration': 'category-specific normalCAL85percent order statistic', 'oneSharedBackbone': True, 'heatmapIsConfirmedDefectLocalization': False})
        self.metal_policy = hashed({'version': 'safelog-metal-plate-calibration-v1', 'category': 'mpdd_metal_plate', 'parentEquipmentPolicySha256': self.equipment_policy['sha256'],
            'neuralModelCount': 1, 'sharedBackboneCount': 1, 'neuralParametersChanged': False, 'photoScore': 'same mean top1percent reconstruction residual',
            'normalCALCount': 20, 'normalCALAlpha': .20, 'normalCALRank': 17, 'pixelThresholdChanged': False, 'heatmapChanged': False, 'TESTThresholdTuning': False})
        self.reference_policies = {}
        for category, count in state['reference_counts'].items():
            point = self.points[category]
            self.reference_policies[category] = hashed({'version': 'safelog-equipment-reference-budget-v2', 'category': category,
                'referenceFITPatches': count, 'referenceStorageDtype': 'float16', 'referenceDistance': 'exact nearest cosine FP32 chunks512/2048',
                'featureSource': 'same one MobileNet low40/high960 branch normalization; avgpool3 stride1 count_include_padFalse',
                'photoScore': 'mean top1percent of bilinear256 normal-reference distances', 'normalCALCount': point['normal_CAL_photos'],
                'normalCALAlpha': point['photo_normalCAL_alpha'], 'normalCALRank': point['photo_normalCAL_rank'], 'neuralModelCount': 1, 'sharedBackboneCount': 1,
                'backbonePassesPerPhoto': 1, 'neuralParametersChanged': False, 'photoScoreSource': 'normal_reference', 'heatmapSource': 'original_reconstruction',
                'pixelThresholdChanged': False, 'originalLocalizationChanged': False, 'referenceHeatmapIsEvaluatedLocalization': False,
                'TESTQuantileThresholdTuning': False, 'candidateSelectionUsedKnownTEST': True, 'adaptiveBenchmarkDevelopment': True, 'independentFieldValidation': False})
        point = self.points['ad_screw']
        if point.get('photo_normalCAL_alpha') != .15 or point.get('photo_normalCAL_rank') != 22: raise ValueError('Native screw normalCAL24 rank22 required')
        self.native_policy = hashed({'version': 'safelog-native-screw-budget-v7', 'category': 'ad_screw', 'viewSize': [768, 256],
            'preprocess': 'input-only Otsu dark largest8connected at256thumbnail; foreground.02to.60; PCA axis ratio>=3; wider head left; isotropic affine12percent margin; median border fill',
            'referenceNormalFITPhotos': 96, 'referenceFITPatches': state['native_specs']['ad_screw']['patches'], 'referenceStorageDtype': 'float16',
            'featureSource': 'same low40/high960 branch normalized patch features, avgpool3 stride1 count_include_padFalse',
            'referenceDistance': 'exact nearest cosine FP32 chunks2048/8192', 'photoScore': 'mean top1percent of native256x768 bilinear cosine distance',
            'normalCALCount': 24, 'normalCALAlpha': .15, 'normalCALRank': 22, 'neuralModelCount': 1, 'sharedBackboneCount': 1,
            'backbonePassesPerPhoto': 2, 'neuralParametersChanged': False, 'original256LocationChanged': False, 'nativeHeatmapUsesInputOnlyAffine': True,
            'referenceHeatmapIsEvaluatedLocalization': False, 'unknownInputQualityAction': '422_manual_review_required', 'TESTQuantileThresholdTuning': False,
            'candidateSelectionUsedKnownTEST': True, 'adaptiveBenchmarkDevelopment': True, 'independentFieldValidation': False})

        self.wear_point = state['wear_operating_point']
        self.wear_policy = hashed({'version': 'safelog-bsclass-supervised-v8', 'category': WEAR_CATEGORY,
            'input': 'cropped ball-screw surface; RGB bicubic256 ImageNet normalized',
            'descriptor': 'same backbone low40/high960 channel mean/population_std/max3000; FP16 roundtrip matching study feature cache; FIT-only standardization clamp[-8,8]',
            'trainedHead': state['wear_variant'], 'selectedEpochByValidation': self.wear_point['selected_epoch_by_validation'],
            'threshold': self.wear_point['photo_threshold'], 'thresholdSource': 'labelled calibration minimax FPR/FNR; frozen before reserved_TEST',
            'actualOptimizerUpdatesAcrossRegisteredCandidates': state['wear_actual_optimizer_updates'],
            'realLabelFITPhotos': 15333, 'validationPhotos': 1653, 'calibrationPhotos': 1613, 'reservedTESTPhotos': 3236,
            'publisherRotationAndNearDuplicatesGrouped': True, 'physicalSpindleOrAcquisitionGroupsVerified': False,
            'backbonePassesPerPhoto': 1, 'neuralModelCount': 1, 'sharedBackboneCount': 1, 'oldBackboneAndHeadsChanged': False,
            'freshReservedTESTFirstScoring': True, 'localizationAvailable': False, 'internalMechanicalFailureAssessed': False, 'independentFieldValidation': False})
        self.surface_policies = {}
        for category, spec in self.surface_specs.items():
            point = self.points[category]
            self.surface_policies[category] = hashed({'version': 'safelog-industrial-surface-reference-v9', 'category': category,
                'method': spec['method'], 'normalFITPhotos': state['surface_reference_fit'][category]['normal_FIT_photos'],
                'referencePatches': spec['patches'], 'referenceStorage': 'float16', 'distance': 'exact cosine FP32 chunks2048/8192',
                'featureSource': 'same shared low40/high960 branch normalized avgpool3 features',
                'photoScore': 'mean top1percent bilinear view evidence', 'normalCALCount': point['normal_CAL_photos'],
                'normalCALAlpha': point['photo_normalCAL_alpha'], 'normalCALRank': point['photo_normalCAL_rank'],
                'threshold': point['photo_threshold'], 'backbonePassesPerPhoto': 1 if spec['method']=='pooled256' else 2,
                'neuralModelCount': 1, 'sharedBackboneCount': 1, 'neuralParametersChanged': False, 'originalLocalizationChanged': False,
                'referenceHeatmapIsEvaluatedLocalization': False, 'candidateSelectionUsedKnownTEST': True, 'independentFieldValidation': False})

    def analyze_equipment(self, category, encoded):
        if category == WEAR_CATEGORY: return self.analyze_wear(encoded)
        if category not in self.surface_specs: return super().analyze_equipment(category, encoded)
        from .industrial_surface_reference import surface_view
        rgb, raw = decode(encoded); spec = self.surface_specs[category]; view, geometry = surface_view(rgb, spec['method'])
        with self.lock, torch.inference_mode(): location, reference, score_tensor = self.model.forward_surface(prepare(rgb), normalized(view)[None], category)
        score = float(score_tensor[0]); point = self.points[category]; threshold = point['photo_threshold']; current = self.surface_policies[category]
        if spec['method'] == 'aspect512':
            heat = Image.open(io.BytesIO(base64.b64decode(png(reference[0], view.size)))).copy()
            x, y = geometry['offset']; w, h = round(rgb.width*geometry['scale']), round(rgb.height*geometry['scale'])
            mapped = heat.crop((x, y, x+w, y+h)).resize(rgb.size, Image.Resampling.BILINEAR)
        else: mapped = Image.open(io.BytesIO(base64.b64decode(png(reference[0], rgb.size)))).copy()
        benchmark = self.evaluation.get(category, {})
        return {'schema': 'safelog.industrial-surface-reference.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': category,
            'classResults': [{'className': category, 'score': score, 'threshold': threshold, 'candidateDetected': score > threshold}], 'candidateDetected': score > threshold,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'original_reconstruction', 'heatmapIsConfirmedDefectLocalization': False,
            'referenceHeatmapPngBase64': encode_png(mapped), 'referenceHeatmapIsEvaluatedLocalization': False, 'nativeViewGeometry': geometry,
            'photoScoreSource': 'industrial_surface_normal_reference', 'scoreIsCalibratedProbability': False,
            'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'pooled_photo_category_on_known_TEST', 'defectGroupBenchmark': benchmark.get('by_defect', {}),
            'allDefectGroupsWithin20': bool(benchmark.get('by_defect')) and all(v['fnr'] <= .20 for v in benchmark['by_defect'].values()),
            'inferencePolicyVersion': current['version'], 'inferencePolicySha256': current['sha256'], 'backbonePassesPerPhoto': current['backbonePassesPerPhoto'],
            'normalCALCount': current['normalCALCount'], 'normalCALAlpha': current['normalCALAlpha'], 'normalCALRank': current['normalCALRank'],
            'evidenceScope': 'specified_metal_component_visible_appearance', 'originalLocalizationChanged': False,
            'candidateSelectionUsedKnownTEST': True, 'adaptiveBenchmarkDevelopment': True, 'mechanicalFailureState': 'not_assessed'}

    def analyze_wear(self, encoded):
        rgb, raw = decode(encoded)
        if min(rgb.size) < 64: raise ValueError('Ball-screw surface crop must be at least64 pixels; review input')
        with self.lock, torch.inference_mode():
            descriptor = wear_descriptor(self.model.backbone(prepare(rgb))).half().float()
            score = float(self.model.forward_wear_descriptor(descriptor).sigmoid()[0])
        threshold = self.wear_point['photo_threshold']; benchmark = self.evaluation.get(WEAR_CATEGORY, {})
        return {'schema': 'safelog.ball-screw-pitting.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': WEAR_CATEGORY,
            'classResults': [{'className': WEAR_CATEGORY, 'score': score, 'threshold': threshold, 'candidateDetected': score > threshold}], 'candidateDetected': score > threshold,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'scoreIsCalibratedProbability': False, 'localizationAvailable': False, 'heatmapIsConfirmedDefectLocalization': False,
            'evidenceScope': 'cropped_ball_screw_visible_pitting', 'requiredInput': 'cropped_ball_screw_surface', 'inputDomainAutoVerified': False,
            'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'grouped_cropped_photo_reserved_TEST_physical_spindle_groups_unverified',
            'inferencePolicyVersion': self.wear_policy['version'], 'inferencePolicySha256': self.wear_policy['sha256'], 'backbonePassesPerPhoto': 1,
            'freshReservedTESTFirstScoring': True, 'physicalSpindleOrAcquisitionGroupsVerified': False, 'mechanicalFailureState': 'not_assessed'}

    def analyze(self, task, encoded, category=None):
        if task == WEAR_CATEGORY or (task == 'equipment_surface' and category == WEAR_CATEGORY):
            result = self.analyze_wear(encoded)
            result.update(neuralModelCount=1, sharedBackboneCount=1, modelSha256=self.digest, trainedDatasetSources=self.sources,
                otherTasksEvaluated=False, humanConfirmationRequired=True, fieldPerformanceValidated=False, researchPrototype=True)
        else: result = super().analyze(task, encoded, category)
        result['modelName'] = ARCH
        return result


class Context(BaseModel):
    inspectionTask:str=Field(min_length=1,max_length=80)
    equipmentCategory:Optional[str]=Field(default=None,max_length=80)
    promptVersion:str=Field(min_length=1,max_length=80)
    @model_validator(mode='after')
    def valid_task(self):
        if self.inspectionTask not in (*TASKS,'equipment_surface',*ALL_CATEGORIES):raise ValueError('Supported task required')
        if self.inspectionTask=='equipment_surface' and self.equipmentCategory not in ALL_CATEGORIES:raise ValueError('Explicit equipment category required')
        if self.inspectionTask in ALL_CATEGORIES and self.equipmentCategory not in (None,self.inspectionTask):raise ValueError('Category alias and category must agree')
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


def create_industrial_app(engine=None):
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
        'benchmarkStage':engine.evaluation.get(c,{}).get('stage_gate',{'state':'not_evaluated','ready_for_next_stage':False}) if engine else {'state':'not_loaded','ready_for_next_stage':False}} for c in ALL_CATEGORIES]}
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
        if result.get('equipmentCategory') == WEAR_CATEGORY: result.update(confidence=raw, confidenceMethod='sigmoid_score_not_calibrated_probability')
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



def create_app(engine=None):
    app = create_industrial_app(engine)
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) not in ('/v1/equipment-categories', '/v1/inference-policy')]
    @app.get('/v1/inference-policy')
    def policies():
        if engine is None: raise HTTPException(503, 'Model not loaded')
        return {'legacyBelt': engine.inference_policy, 'equipment': engine.equipment_policy, 'metalPlate': engine.metal_policy,
            'referencePhoto': engine.reference_policies, 'nativeComponents': {'ad_screw': engine.native_policy},
            'supervisedWear': {WEAR_CATEGORY: engine.wear_policy}, 'surfaceReference': engine.surface_policies}
    @app.get('/v1/equipment-categories')
    def categories():
        values = []
        for category in ALL_CATEGORIES:
            benchmark = engine.evaluation.get(category, {}) if engine else {}
            row = {'id': category, 'label': LABELS[category], 'trained': bool(engine and (category == WEAR_CATEGORY or category in engine.points)),
                'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False})}
            if category == 'ad_screw' or (engine and category in engine.surface_specs):
                groups = benchmark.get('by_defect', {})
                row.update(benchmarkStageScope='pooled_photo_category_on_known_TEST', defectGroupBenchmark=groups,
                    allDefectGroupsWithin20=bool(groups) and all(v['fnr'] <= .20 for v in groups.values()))
            if category == WEAR_CATEGORY: row.update(requiredInput='cropped_ball_screw_surface', localizationAvailable=False,
                benchmarkStageScope='grouped_cropped_photo_reserved_TEST_physical_spindle_groups_unverified')
            values.append(row)
        return {'categories': values}
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation', required=True); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_app(IndustrialEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)

