"""Selected normalCAL10/05 photo policies, with prior location outputs retained."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from threading import RLock
import torch
from .equipment_reference_photo import ARCH, REFERENCE_CATEGORIES, ReferencePhotoMain
from .equipment_reference_server import ReferencePhotoEngine, create_reference_app, hashed
from .equipment_expansion import CATEGORIES
from .belt_tiles import policy


class ReferenceBudgetEngine(ReferencePhotoEngine):
    def __init__(self, checkpoint, expected, evaluation=None):
        raw = Path(checkpoint).read_bytes(); self.digest = hashlib.sha256(raw).hexdigest()
        if self.digest != expected: raise ValueError('Explicit prototype checkpoint SHA mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True); counts = state.get('reference_counts', {})
        if (state.get('architecture') != ARCH or not state.get('one_shared_backbone') or
                state.get('equipment_categories') != list(CATEGORIES) or not counts or
                any(c not in REFERENCE_CATEGORIES for c in counts) or not state.get('reference_budget_protocol_sha256') or
                state.get('metal_refinement_variant') != 'calibration_only20' or state.get('normalCAL_rank') != 17):
            raise ValueError('Selected one-backbone normal-reference budget checkpoint with metal20 required')
        self.model = ReferencePhotoMain(counts).eval(); self.model.load_state_dict(state['state_dict'], strict=True)
        self.lock = RLock(); self.thresholds = state['thresholds']; self.points = state['equipment_operating_points']
        self.sources = state['trained_dataset_sources']; self.mode = 'global_tile_mean'; self.inference_policy = policy(self.mode)
        self.evaluation = {}
        if evaluation:
            info = json.loads(Path(evaluation).read_text(encoding='utf-8'))
            if info['checkpoint_sha256'] != self.digest: raise ValueError('Evaluation must belong to this exact checkpoint')
            self.evaluation = info['categories']
        self.equipment_policy = hashed({'version': 'safelog-equipment-expansion-v1', 'preprocess': 'bicubic RGB256 square, ImageNet normalization',
            'score': 'mean top1percent raw cosine residual', 'normal_calibration': 'category-specific normalCAL85percent order statistic',
            'oneSharedBackbone': True, 'heatmapIsConfirmedDefectLocalization': False})
        self.metal_policy = hashed({'version': 'safelog-metal-plate-calibration-v1', 'category': 'mpdd_metal_plate',
            'parentEquipmentPolicySha256': self.equipment_policy['sha256'], 'neuralModelCount': 1, 'sharedBackboneCount': 1, 'neuralParametersChanged': False,
            'photoScore': 'same mean top1percent reconstruction residual', 'normalCALCount': 20, 'normalCALAlpha': .20, 'normalCALRank': 17,
            'pixelThresholdChanged': False, 'heatmapChanged': False, 'TESTThresholdTuning': False})
        self.reference_policies = {}
        for category, count in counts.items():
            point = self.points[category]; alpha = point.get('photo_normalCAL_alpha')
            if point.get('photo_score_method') != 'normal_reference_top1percent' or alpha not in (.10, .05):
                raise ValueError('Explicit selected normalCAL10/05 reference-photo policy required')
            self.reference_policies[category] = hashed({'version': 'safelog-equipment-reference-budget-v2', 'category': category,
                'referenceFITPatches': count, 'referenceStorageDtype': 'float16', 'referenceDistance': 'exact nearest cosine FP32 chunks512/2048',
                'featureSource': 'same one MobileNet low40/high960 branch normalization; avgpool3 stride1 count_include_padFalse',
                'photoScore': 'mean top1percent of bilinear256 normal-reference distances', 'normalCALCount': point['normal_CAL_photos'],
                'normalCALAlpha': alpha, 'normalCALRank': point['photo_normalCAL_rank'], 'neuralModelCount': 1, 'sharedBackboneCount': 1,
                'backbonePassesPerPhoto': 1, 'neuralParametersChanged': False, 'photoScoreSource': 'normal_reference',
                'heatmapSource': 'original_reconstruction', 'pixelThresholdChanged': False, 'originalLocalizationChanged': False,
                'referenceHeatmapIsEvaluatedLocalization': False, 'TESTQuantileThresholdTuning': False,
                'candidateSelectionUsedKnownTEST': True, 'adaptiveBenchmarkDevelopment': True, 'independentFieldValidation': False})

    def analyze_equipment(self, category, encoded):
        result = super().analyze_equipment(category, encoded)
        if category in self.reference_policies:
            result.update(normalCALAlpha=self.points[category]['photo_normalCAL_alpha'], adaptiveBenchmarkDevelopment=True,
                          candidateSelectionUsedKnownTEST=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--evaluation'); parser.add_argument('--port', type=int, default=8775); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import uvicorn
    uvicorn.run(create_reference_app(ReferenceBudgetEngine(args.checkpoint, args.sha256, args.evaluation)), host='127.0.0.1', port=args.port)
