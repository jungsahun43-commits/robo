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
from .ksdd_photo_only import KSDDPhotoMain, VERSION
from .native_component_views import normalized
from .native_component_server import encode_png
from .equipment_expansion import CATEGORIES, photo_score
from .equipment_expansion_server import prepare
from .equipment_reference_server import hashed, png
from .unified_main_server import decode
from .belt_tiles import policy


class KSDDPhotoEngine(IndustrialEngine):
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
        self.model = KSDDPhotoMain(state['reference_counts'], state['native_specs'], state['wear_variant'], self.surface_specs)
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

    def analyze_equipment(self, category, encoded):
        if category != CATEGORY: return super().analyze_equipment(category, encoded)
        rgb, raw = decode(encoded)
        if min(rgb.size) < 64: raise ValueError('Surface closeup must be at least64 pixels; review input')
        view, _, valid, geometry = ksdd_view(rgb)
        with self.lock, torch.inference_mode():
            location, auxiliary, value = self.model.forward_ksdd_photo(prepare(rgb), normalized(view)[None], valid[None])
        score = float(value[0]); threshold = self.ksdd_point['photo_threshold']
        restored = restore_map(auxiliary[0], geometry).clamp(0, 1)
        auxiliary_image = Image.fromarray((restored.numpy()*255).astype('uint8'))
        mask_image = Image.fromarray((restored.numpy() > self.ksdd_point['pixel_threshold']).astype('uint8')*255)
        benchmark = self.evaluation.get(CATEGORY, {})
        return {'schema': 'safelog.ksdd-supervised-photo-only.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': category,
            'classResults': [{'className': category, 'score': score, 'threshold': threshold, 'candidateDetected': score > threshold}], 'candidateDetected': score > threshold,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'retained_original_reconstruction', 'originalLocalizationChanged': False,
            'heatmapIsConfirmedDefectLocalization': False, 'originalPixelThreshold': self.points[CATEGORY]['pixel_threshold'],
            'supervisedAuxiliaryHeatmapPngBase64': encode_png(auxiliary_image), 'supervisedAuxiliaryMaskPngBase64': encode_png(mask_image),
            'auxiliarySegmentationThreshold': self.ksdd_point['pixel_threshold'], 'auxiliaryLocalizationWorseFNR': True, 'auxiliaryIsPrimaryLocationOutput': False,
            'auxiliarySupervisedOriginalPixelMetrics': benchmark.get('auxiliary_supervised_pixel_metrics', {}),
            'primaryOriginalCoordinatePixelMetrics': benchmark.get('original_coordinate_parent_pixel_metrics', {}),
            'photoScoreSource': 'real_mask_supervised_photo_only', 'scoreIsCalibratedProbability': False, 'backbonePassesPerPhoto': 2,
            'evidenceScope': 'publisher_inspected_component_visible_surface', 'mechanicalFailureState': 'not_assessed', 'nativeViewGeometry': geometry,
            'inferencePolicyVersion': VERSION, 'inferencePolicySha256': self.ksdd_policy['sha256'],
            'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'known_TEST_photo_only_original_location_preserved',
            'knownTESTUsedForFollowupDesignAndSelection': True, 'adaptiveBenchmarkDevelopment': True, 'modelExtensionVersion': VERSION}

    def analyze(self, task, encoded, category=None):
        result = super().analyze(task, encoded, category); result['modelName'] = VERSION
        return result


def create_app(engine=None):
    app = create_previous_app(engine)
    old_policies = next(r.endpoint for r in app.router.routes if getattr(r, 'path', None) == '/v1/inference-policy')
    old_categories = next(r.endpoint for r in app.router.routes if getattr(r, 'path', None) == '/v1/equipment-categories')
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) not in ('/v1/inference-policy', '/v1/equipment-categories')]
    @app.get('/v1/inference-policy')
    def policies():
        value = old_policies(); value['supervisedPhoto'] = {CATEGORY: engine.ksdd_policy}; return value
    @app.get('/v1/equipment-categories')
    def categories():
        value = old_categories()
        for row in value['categories']:
            if row['id'] == CATEGORY: row.update(photoScoreSource='real_mask_supervised_photo_only', originalLocalizationChanged=False,
                auxiliaryLocalizationWorseFNR=True, benchmarkStageScope='known_TEST_photo_only_original_location_preserved')
        return value
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation', required=True); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_app(KSDDPhotoEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)
