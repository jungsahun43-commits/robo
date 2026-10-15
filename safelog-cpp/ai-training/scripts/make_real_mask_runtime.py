"""Extend the immutable prior constructor without changing its original policies."""
import inspect
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.loco_context_server import ContextEngine
from scripts import study_real_mask_transfer as study
from scripts.fetch_equipment_expansion import sha


def main():
    p=study.validate();prior=ROOT/'safelog_ai/loco_context_server.py'
    assert sha(prior)==p['legacy_source_sha256']['safelog_ai/loco_context_server.py']
    target=ROOT/'safelog_ai/real_mask_server.py';assert not target.exists()
    header=prior.read_text(encoding='utf-8').split('class ContextEngine(',1)[0]
    header+='from .loco_context_server import ContextEngine, create_app as create_context_app\n'
    header+='from .magnetic_surface import RealMaskMain, surface_view, photo_score as transfer_photo_score, VERSION as TRANSFER_VERSION\n'
    header+='from .industrial_surface_reference import IndustrialSurfaceMain\n\nclass RuntimeRealMaskMain(RealMaskMain):\n    forward_surface = IndustrialSurfaceMain.forward_surface\n    forward_real_mask = RealMaskMain.forward_surface\n'
    constructor=inspect.getsource(ContextEngine.__init__).replace('self.model = ContextMain(', 'self.model = RuntimeRealMaskMain(')
    extra='''        if state.get('transfer_surface_methods') != {'mpdd_bracket_black': 'max_transfer_local'}:
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
'''
    methods='''
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
'''
    source=header+'\nclass RealMaskEngine(ContextEngine):\n'+constructor.rstrip()+'\n'+extra+methods
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    print({'runtime_written':str(target),'prior_source_SHA_preserved':sha(prior)},flush=True)


if __name__=='__main__':main()
