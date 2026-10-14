"""One-checkpoint reference photo presence plus preserved location outputs."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
from PIL import Image
import torch
from .equipment_reference_photo import ARCH, REFERENCE_CATEGORIES, ReferencePhotoMain
from .equipment_expansion import CATEGORIES, photo_score
from .equipment_expansion_server import prepare
from .metal_calibration_server import MetalCalibrationEngine, create_metal_app
from .unified_main_server import decode
from .belt_tiles import policy


def hashed(setting):
    return {**setting, 'sha256': hashlib.sha256(json.dumps(setting, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def png(heat, size):
    image = Image.fromarray((heat.clamp(0, 2).numpy()/2*255).astype('uint8')).resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


class ReferencePhotoEngine(MetalCalibrationEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw = Path(checkpoint).read_bytes(); self.digest = hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit prototype checkpoint SHA mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        counts = state.get('reference_counts', {})
        if (state.get('architecture') != ARCH or not state.get('one_shared_backbone') or
                state.get('equipment_categories') != list(CATEGORIES) or not counts or
                any(c not in REFERENCE_CATEGORIES for c in counts) or
                state.get('metal_refinement_variant') != 'calibration_only20' or state.get('normalCAL_rank') != 17):
            raise ValueError('One-backbone selected reference-photo checkpoint with retained metal20 required')
        self.model = ReferencePhotoMain(counts).eval(); self.model.load_state_dict(state['state_dict'], strict=True)
        self.lock = RLock(); self.thresholds = state['thresholds']; self.points = state['equipment_operating_points']
        self.sources = state['trained_dataset_sources']; self.mode = 'global_tile_mean'; self.inference_policy = policy(self.mode)
        self.evaluation = {}
        if evaluation:
            info = json.loads(Path(evaluation).read_text(encoding='utf-8'))
            if info['checkpoint_sha256'] != self.digest: raise ValueError('Evaluation must belong to this exact checkpoint')
            self.evaluation = info['categories']
        # Preserve the exact prior policies for all unaffected categories.
        self.equipment_policy = hashed({'version': 'safelog-equipment-expansion-v1', 'preprocess': 'bicubic RGB256 square, ImageNet normalization',
            'score': 'mean top1percent raw cosine residual', 'normal_calibration': 'category-specific normalCAL85percent order statistic',
            'oneSharedBackbone': True, 'heatmapIsConfirmedDefectLocalization': False})
        self.metal_policy = hashed({'version': 'safelog-metal-plate-calibration-v1', 'category': 'mpdd_metal_plate',
            'parentEquipmentPolicySha256': self.equipment_policy['sha256'], 'neuralModelCount': 1, 'sharedBackboneCount': 1, 'neuralParametersChanged': False,
            'photoScore': 'same mean top1percent reconstruction residual', 'normalCALCount': 20, 'normalCALAlpha': .20, 'normalCALRank': 17,
            'pixelThresholdChanged': False, 'heatmapChanged': False, 'TESTThresholdTuning': False})
        self.reference_policies = {}
        for category, count in counts.items():
            point = self.points[category]
            if point.get('photo_score_method') != 'normal_reference_top1percent' or point.get('photo_normalCAL_alpha') != .20:
                raise ValueError('Explicit normal reference presence score and normalCAL20 required')
            self.reference_policies[category] = hashed({'version': 'safelog-equipment-reference-photo-v1', 'category': category,
                'referenceFITPatches': count, 'referenceStorageDtype': 'float16', 'referenceDistance': 'exact nearest cosine FP32 chunks512/2048',
                'featureSource': 'same one MobileNet low40/high960 branch normalization; avgpool3 stride1 count_include_padFalse',
                'photoScore': 'mean top1percent of bilinear256 normal-reference distances', 'normalCALCount': point['normal_CAL_photos'],
                'normalCALAlpha': .20, 'normalCALRank': point['photo_normalCAL_rank'], 'neuralModelCount': 1, 'sharedBackboneCount': 1,
                'backbonePassesPerPhoto': 1, 'neuralParametersChanged': False, 'photoScoreSource': 'normal_reference',
                'heatmapSource': 'original_reconstruction', 'pixelThresholdChanged': False, 'originalLocalizationChanged': False,
                'referenceHeatmapIsEvaluatedLocalization': False, 'TESTThresholdTuning': False})

    def analyze_equipment(self, category, encoded):
        if category not in self.reference_policies: return super().analyze_equipment(category, encoded)
        rgb, raw = decode(encoded)
        with self.lock, torch.inference_mode(): location, reference = self.model.forward_equipment_evidence(prepare(rgb), category)
        score = float(photo_score(reference)[0]); point = self.points[category]; threshold = point['photo_threshold']; candidate = score > threshold
        current = self.reference_policies[category]
        return {'schema': 'safelog.equipment-reference-photo.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': category,
            'classResults': [{'className': category, 'score': score, 'threshold': threshold, 'candidateDetected': candidate}], 'candidateDetected': candidate,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'original_reconstruction', 'heatmapIsConfirmedDefectLocalization': False,
            'referenceHeatmapPngBase64': png(reference[0], rgb.size), 'referenceHeatmapIsEvaluatedLocalization': False,
            'photoScoreSource': 'normal_reference', 'scoreIsCalibratedProbability': False, 'evidenceScope': 'inspected_component_external_appearance',
            'originalLocalizationChanged': False, 'originalLocalizationPhotoScore': float(photo_score(location)[0]),
            'inferencePolicyVersion': current['version'], 'inferencePolicySha256': current['sha256'], 'modelExtensionVersion': ARCH,
            'benchmarkStage': self.evaluation.get(category, {}).get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'normalCALCount': point['normal_CAL_photos'], 'normalCALAlpha': .20, 'normalCALRank': point['photo_normalCAL_rank'],
            'referenceFITPatches': self.model.reference_counts[category], 'backbonePassesPerPhoto': 1, 'mechanicalFailureState': 'not_assessed'}


def create_reference_app(engine=None):
    app = create_metal_app(engine)
    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', None) != '/v1/inference-policy']
    @app.get('/v1/inference-policy')
    def policies():
        if engine is None:
            from fastapi import HTTPException
            raise HTTPException(503, 'Model not loaded')
        return {'legacyBelt': engine.inference_policy, 'equipment': engine.equipment_policy,
                'metalPlate': engine.metal_policy, 'referencePhoto': engine.reference_policies}
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation'); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_reference_app(ReferencePhotoEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)
