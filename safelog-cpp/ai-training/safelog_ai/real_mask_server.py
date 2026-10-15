"""Real-mask photo presence and exact retained location output, one checkpoint."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
import numpy as np
from PIL import Image
import torch
from .industrial_batch_server import IndustrialEngine, create_app as create_previous_app
from .industrial_supervised import WEAR_CATEGORY
from .ksdd_supervised import ARCH as KSDD_ARCH, CATEGORY, ksdd_view, restore_map
from .ksdd_photo_only import VERSION
from .loco_context import ContextMain, CATEGORY as CONTEXT_CATEGORY, VERSION as CONTEXT_VERSION, context_view
from .native_component_views import normalized
from .native_component_server import encode_png
from .equipment_expansion import CATEGORIES, photo_score
from .equipment_expansion_server import prepare
from .equipment_reference_server import hashed, png
from .unified_main_server import decode
from .belt_tiles import policy


from .loco_context_server import ContextEngine, create_app as create_context_app
from .magnetic_surface import RealMaskMain, surface_view, photo_score as transfer_photo_score, VERSION as TRANSFER_VERSION
from .industrial_surface_reference import IndustrialSurfaceMain


class RuntimeRealMaskMain(RealMaskMain):
    # Preserve the existing brown/white reference method signature exactly.
    forward_surface = IndustrialSurfaceMain.forward_surface
    forward_real_mask = RealMaskMain.forward_surface

class RealMaskEngine(ContextEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw = Path(checkpoint).read_bytes(); self.digest = hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit research checkpoint SHA mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (state.get('architecture') != KSDD_ARCH or not state.get('one_shared_backbone') or
            state.get('equipment_categories') != list(CATEGORIES) or state.get('wear_category') != WEAR_CATEGORY or
            state.get('native_screw_budget_variant') != 'normalCAL15' or not state.get('wear_protocol_sha256')):
            raise ValueError('Selected one-backbone supervised industrial checkpoint required')
        self.surface_specs = state.get('surface_reference_specs', {})
        if not state.get('ksdd_photo_only') or not state.get('ksdd_photo_only_protocol_sha256'):
            raise ValueError('Explicit photo-only composition preserving original location required')
        self.model = RuntimeRealMaskMain(state['reference_counts'], state['native_specs'], state['wear_variant'], self.surface_specs)
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

        self.ksdd_point = state['ksdd_operating_point']
        self.ksdd_policy = hashed({'version': VERSION, 'category': CATEGORY,
            'photoScore': 'real-mask supervised head, mean top32 native valid sigmoid pixels',
            'photoThreshold': self.ksdd_point['photo_threshold'], 'thresholdSource': 'labelled CAL490; frozen before v10 TEST, unchanged in v11',
            'input': 'RGB aspect-preserving256x704 letterbox127 ImageNet normalization',
            'features': 'same accepted backbone low40/high960 FP16 roundtrip; high960to32 and low40 learned mask fusion',
            'actualPriorRealMaskOptimizerUpdates': 2560, 'selectedEpochByValidation': 13, 'selectedEpochUpdates': 832,
            'newOptimizerUpdatesThisComposition': 0, 'backbonePassesPerPhoto': 2, 'neuralModelCount': 1, 'sharedBackboneCount': 1,
            'primaryHeatmap': 'exact original256 reconstruction, original source coordinates, old threshold retained',
            'originalPixelThreshold': self.points[CATEGORY]['pixel_threshold'], 'originalLocalizationChanged': False,
            'auxiliarySupervisedMaskNotPrimary': True, 'auxiliaryLocalizationWorseFNR': True,
            'knownTESTUsedForFollowupDesignAndSelection': True, 'independentFieldValidation': False})

        if state.get('context_version') != CONTEXT_VERSION or state.get('context_selected_method') not in ('context_only', 'max_context_local'):
            raise ValueError('Explicit selected normal-trained context research checkpoint required')
        self.context_method = state['context_selected_method']; self.context_point = state['context_operating_point']
        self.context_policy = hashed({'version': CONTEXT_VERSION, 'category': CONTEXT_CATEGORY,
            'view': 'isotropic512x256 letterbox127 RGB ImageNet normalized',
            'target': 'same backbone low40/high960 unit features pooled4x8, FP16 roundtrip matching frozen study',
            'head': 'normalFIT-only global128 bottleneck reconstruction with FIT-only channel standardization clamp[-8,8]',
            'method': self.context_method, 'operatingPoint': self.context_point,
            'thresholdSource': 'normalCAL44; both score candidates fixed before TEST',
            'actualOptimizerUpdates': state['context_actual_optimizer_updates'], 'selectedEpochByNormalValidation': state['context_selected_epoch'],
            'normalFITPhotos': 199, 'normalValidationPhotos': 55, 'normalCALPhotos': state['context_normal_CAL_photos'],
            'newDefectOrObjectRelationLabels': 0, 'semanticClampWireRelationsVerified': False,
            'backbonePassesPerPhoto': 2, 'sharedBackboneCount': 1, 'neuralModelCount': 1,
            'originalLocalizationChanged': False, 'originalPixelThreshold': self.points[CONTEXT_CATEGORY]['pixel_threshold'],
            'knownTESTUsedForCandidateSelection': True, 'independentFieldValidation': False})
        if state.get('transfer_surface_methods') != {'mpdd_bracket_black': 'max_transfer_local'}:
            raise ValueError('Only the validated research black-bracket composition is enabled')
        self.transfer_methods = state['transfer_surface_methods']; self.transfer_points = state['transfer_surface_points']
        self.transfer_policies = {}
        for c, method in self.transfer_methods.items():
            point = self.transfer_points[c]
            self.transfer_policies[c] = hashed({'version': TRANSFER_VERSION, 'category': c, 'method': method,
                'view': '512 square isotropic letterbox127, ImageNet normalization', 'featurePrecision': 'FP16 roundtrip then FP32',
                'photoScore': 'mean top32 valid native sigmoid pixels, max-fused with preserved local score',
                'point': point, 'normalCALPhotos': 30, 'normalCALAlpha': .15,
                'realMaskFITDefects': 217, 'normalFITPhotos': 499, 'actualOptimizerUpdates': state['transfer_training_actual_updates'],
                'selectedEpochByLabelledValidation': state['transfer_selected_epoch'], 'selectedEpochUpdates': state['transfer_selected_epoch_updates'],
                'sourceDataset': 'author magnetic tiles, academic research only, cite author paper',
                'sourceHeldoutPhotoBenchmarkPassed': False, 'sourcePixelFNRAtHalf': .8393472790013606,
                'originalLocalizationChanged': False, 'transferIsValidatedPixelLocalization': False,
                'neuralModelCount': 1, 'sharedBackboneCount': 1, 'backbonePassesPerPhoto': 2,
                'knownTESTUsedForCandidateSelection': True, 'independentFieldValidation': False})

    def analyze_equipment(self, category, encoded):
        if category not in self.transfer_methods: return super().analyze_equipment(category, encoded)
        rgb, raw = decode(encoded); view, _, valid, geometry = surface_view(rgb)
        with self.lock, torch.inference_mode():
            location = self.model.forward_equipment(prepare(rgb), category)
            probability = self.model.forward_real_mask(normalized(view)[None]).sigmoid()
            raw_score = float(transfer_photo_score(probability, valid[None])[0])
        local_score = float(photo_score(location)[0]); point = self.transfer_points[category]
        combined = max(raw_score/point['transfer_divisor'], local_score/point['local_divisor'])
        cut = point['photo_threshold']; benchmark = self.evaluation.get(category, {})
        return {'schema': 'safelog.real-mask-transfer-photo.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': category,
            'classResults': [{'className': category, 'score': combined, 'threshold': cut, 'candidateDetected': combined > cut}],
            'candidateDetected': combined > cut, 'realMaskTransferScore': raw_score, 'localAppearanceScore': local_score,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'retained_original_reconstruction',
            'originalLocalizationChanged': False, 'originalPixelThreshold': self.points[category]['pixel_threshold'],
            'heatmapIsConfirmedDefectLocalization': False, 'transferIsValidatedPixelLocalization': False,
            'photoScoreSource': self.transfer_methods[category], 'scoreIsCalibratedProbability': False,
            'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'known_TEST_partial_photo_improvement_not20percent_pass',
            'defectGroupBenchmark': benchmark.get('by_defect', {}),
            'sourceHeldoutPhotoBenchmarkPassed': False, 'sourceDatasetIsTargetDomain': False,
            'inferencePolicyVersion': TRANSFER_VERSION, 'inferencePolicySha256': self.transfer_policies[category]['sha256'],
            'backbonePassesPerPhoto': 2, 'mechanicalFailureState': 'not_assessed',
            'evidenceScope': 'specified_black_bracket_visible_surface_photo_only', 'nativeViewGeometry': geometry,
            'knownTESTUsedForCandidateSelection': True, 'independentFieldValidation': False}

    def analyze(self, task, encoded, category=None):
        result = super().analyze(task, encoded, category); result['modelName'] = TRANSFER_VERSION
        return result


def create_app(engine=None):
    app = create_context_app(engine)
    old_policies = next(r.endpoint for r in app.router.routes if getattr(r, 'path', None) == '/v1/inference-policy')
    old_categories = next(r.endpoint for r in app.router.routes if getattr(r, 'path', None) == '/v1/equipment-categories')
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) not in ('/v1/inference-policy', '/v1/equipment-categories')]
    @app.get('/v1/inference-policy')
    def policies():
        value = old_policies(); value['realMaskTransfer'] = engine.transfer_policies; return value
    @app.get('/v1/equipment-categories')
    def categories():
        value = old_categories()
        for row in value['categories']:
            if row['id'] in engine.transfer_methods: row.update(photoScoreSource=engine.transfer_methods[row['id']],
                photoThreshold=engine.transfer_points[row['id']]['photo_threshold'], originalLocalizationChanged=False,
                benchmarkStageScope='known_TEST_partial_photo_improvement_not20percent_pass')
        return value
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation', required=True); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_app(RealMaskEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)
