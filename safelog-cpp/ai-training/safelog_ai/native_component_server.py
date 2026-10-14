"""Selected normalCAL15 native screw prototype with preserved prior policies."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
import numpy as np
from PIL import Image
import torch
from .native_component_views import ARCH, NativeComponentMain, component_view, normalized
from .equipment_expansion import CATEGORIES, photo_score
from .equipment_expansion_server import prepare, LABELS
from .equipment_reference_budget_server import ReferenceBudgetEngine
from .equipment_reference_server import hashed, png, create_reference_app
from .unified_main_server import decode
from .belt_tiles import policy


def encode_png(image):
    buffer = io.BytesIO(); image.save(buffer, format='PNG'); return base64.b64encode(buffer.getvalue()).decode()


def map_native_heatmap(heat, original_size, geometry):
    image = Image.fromarray((heat.clamp(0, 2).numpy()/2*255).astype('uint8'))
    affine = np.vstack((np.array(geometry['source_from_view_affine']).reshape(2, 3), [0., 0., 1.]))
    inverse = tuple(np.linalg.inv(affine)[:2].flatten())
    mapped = image.transform(original_size, Image.Transform.AFFINE, inverse, Image.Resampling.BILINEAR, fillcolor=0)
    valid = Image.new('L', image.size, 255).transform(original_size, Image.Transform.AFFINE, inverse, Image.Resampling.NEAREST, fillcolor=0)
    return mapped, valid


class NativeComponentEngine(ReferenceBudgetEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw = Path(checkpoint).read_bytes(); self.digest = hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit prototype checkpoint SHA mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (state.get('architecture') != ARCH or not state.get('one_shared_backbone') or state.get('equipment_categories') != list(CATEGORIES) or
            state.get('native_screw_budget_variant') != 'normalCAL15' or set(state.get('native_specs', {})) != {'ad_screw'} or
            state['native_specs']['ad_screw']['method'] != 'aligned_hilo'):
            raise ValueError('Explicit selected native screw CAL15 one-backbone checkpoint required')
        self.model = NativeComponentMain(state['reference_counts'], state['native_specs']).eval(); self.model.load_state_dict(state['state_dict'], strict=True)
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

    def analyze_equipment(self, category, encoded):
        if category != 'ad_screw': return super().analyze_equipment(category, encoded)
        rgb, raw = decode(encoded); view, geometry = component_view(rgb, category)
        with self.lock, torch.inference_mode(): location, reference, score_tensor, _ = self.model.forward_native(prepare(rgb), normalized(view)[None], category)
        score = float(score_tensor[0]); point = self.points[category]; threshold = point['photo_threshold']; candidate = score > threshold
        mapped, valid = map_native_heatmap(reference[0], rgb.size, geometry)
        return {'schema': 'safelog.native-component.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': category,
            'classResults': [{'className': category, 'score': score, 'threshold': threshold, 'candidateDetected': candidate}], 'candidateDetected': candidate,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'original_reconstruction', 'heatmapIsConfirmedDefectLocalization': False,
            'referenceHeatmapPngBase64': encode_png(mapped), 'referenceValidRegionMaskPngBase64': encode_png(valid),
            'nativeReferenceHeatmapPngBase64': png(reference[0], view.size), 'nativeViewImagePngBase64': encode_png(view), 'nativeViewGeometry': geometry,
            'referenceHeatmapIsEvaluatedLocalization': False, 'photoScoreSource': 'pose_aligned_native_normal_reference', 'scoreIsCalibratedProbability': False,
            'evidenceScope': 'pose_normalized_single_screw_visible_appearance', 'originalLocalizationChanged': False, 'modelExtensionVersion': ARCH,
            'normalCALCount': 24, 'normalCALAlpha': .15, 'normalCALRank': 22, 'backbonePassesPerPhoto': 2, 'mechanicalFailureState': 'not_assessed',
            'inferencePolicyVersion': self.native_policy['version'], 'inferencePolicySha256': self.native_policy['sha256'],
            'benchmarkStage': self.evaluation.get(category, {}).get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'pooled_photo_category_on_known_TEST', 'defectGroupBenchmark': self.evaluation.get(category, {}).get('by_defect', {}),
            'allDefectGroupsWithin20': False,
            'adaptiveBenchmarkDevelopment': True, 'candidateSelectionUsedKnownTEST': True}


def create_native_app(engine=None):
    app = create_reference_app(engine); app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) != '/v1/inference-policy']
    @app.get('/v1/inference-policy')
    def policies():
        if engine is None:
            from fastapi import HTTPException
            raise HTTPException(503, 'Model not loaded')
        return {'legacyBelt': engine.inference_policy, 'equipment': engine.equipment_policy, 'metalPlate': engine.metal_policy,
                'referencePhoto': engine.reference_policies, 'nativeComponents': {'ad_screw': engine.native_policy}}
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) != '/v1/equipment-categories']
    @app.get('/v1/equipment-categories')
    def categories():
        values = []
        for category in CATEGORIES:
            benchmark = engine.evaluation.get(category, {}) if engine else {}
            row = {'id': category, 'label': LABELS[category], 'trained': bool(engine and category in engine.points),
                'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False})}
            if category == 'ad_screw': row.update(benchmarkStageScope='pooled_photo_category_on_known_TEST', defectGroupBenchmark=benchmark.get('by_defect', {}), allDefectGroupsWithin20=False)
            values.append(row)
        return {'categories': values}
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation'); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_native_app(NativeComponentEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)
