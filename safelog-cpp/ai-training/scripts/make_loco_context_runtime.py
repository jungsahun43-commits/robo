"""Generate the v13 adapter while preserving existing policy initialisation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_loco_context as study
from scripts.fetch_equipment_expansion import sha


def main():
    protocol=study.validate();source=ROOT/'safelog_ai/ksdd_photo_only_server.py'
    assert sha(source)==protocol['legacy_source_sha256']['safelog_ai/ksdd_photo_only_server.py']
    path=ROOT/'safelog_ai/loco_context_server.py';assert not path.exists()
    text=source.read_text(encoding='utf-8')
    text=text.replace('from .ksdd_photo_only import KSDDPhotoMain, VERSION','from .ksdd_photo_only import VERSION\nfrom .loco_context import ContextMain, CATEGORY as CONTEXT_CATEGORY, VERSION as CONTEXT_VERSION, context_view')
    text=text.replace('KSDDPhotoEngine','ContextEngine').replace('self.model = KSDDPhotoMain(','self.model = ContextMain(')
    insert='''        if state.get('context_version') != CONTEXT_VERSION or state.get('context_selected_method') not in ('context_only', 'max_context_local'):
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

    def analyze_context(self, encoded):
        rgb, raw = decode(encoded); view, geometry = context_view(rgb)
        with self.lock, torch.inference_mode():
            location, evidence, score = self.model.forward_context_and_location(prepare(rgb), normalized(view)[None])
        global_score = float(score[0]); local_score = float(photo_score(location)[0]); point = self.context_point
        combined = global_score if self.context_method == 'context_only' else max(global_score/point['global_divisor'], local_score/point['local_divisor'])
        threshold = point['photo_threshold']; benchmark = self.evaluation.get(CONTEXT_CATEGORY, {})
        return {'schema': 'safelog.loco-context.v1', 'inspectionTask': 'equipment_surface', 'equipmentCategory': CONTEXT_CATEGORY,
            'classResults': [{'className': CONTEXT_CATEGORY, 'score': combined, 'threshold': threshold, 'candidateDetected': combined > threshold}],
            'candidateDetected': combined > threshold, 'globalContextScore': global_score, 'localAppearanceScore': local_score,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': png(location[0], rgb.size), 'heatmapSource': 'retained_original_reconstruction',
            'originalLocalizationChanged': False, 'originalPixelThreshold': self.points[CONTEXT_CATEGORY]['pixel_threshold'],
            'heatmapIsConfirmedDefectLocalization': False, 'contextIsEvaluatedPixelLocalization': False,
            'photoScoreSource': self.context_method, 'scoreIsCalibratedProbability': False,
            'benchmarkStage': benchmark.get('stage_gate', {'state': 'not_evaluated', 'ready_for_next_stage': False}),
            'benchmarkStageScope': 'known_TEST_photo_incremental_improvement_not20percent_pass',
            'defectGroupBenchmark': benchmark.get('by_defect', {}), 'semanticClampWireRelationsVerified': False,
            'inferencePolicyVersion': CONTEXT_VERSION, 'inferencePolicySha256': self.context_policy['sha256'], 'backbonePassesPerPhoto': 2,
            'mechanicalFailureState': 'not_assessed', 'evidenceScope': 'publisher_two_connector_visible_configuration_only',
            'nativeViewGeometry': geometry, 'knownTESTUsedForCandidateSelection': True, 'independentFieldValidation': False}

'''
    needle='    def analyze_equipment(self, category, encoded):\n'
    assert text.count(needle)==1
    text=text.replace(needle,insert+needle+'        if category == CONTEXT_CATEGORY: return self.analyze_context(encoded)\n')
    text=text.replace("result['modelName'] = VERSION", "result['modelName'] = CONTEXT_VERSION")
    text=text.replace("value['supervisedPhoto'] = {CATEGORY: engine.ksdd_policy}; return value", "value['supervisedPhoto'] = {CATEGORY: engine.ksdd_policy}; value['globalContext'] = {CONTEXT_CATEGORY: engine.context_policy}; return value")
    text=text.replace("        return value\n    return app", "            if row['id'] == CONTEXT_CATEGORY: row.update(photoScoreSource=engine.context_method, photoThreshold=engine.context_point['photo_threshold'], originalLocalizationChanged=False, semanticClampWireRelationsVerified=False)\n        return value\n    return app")
    with path.open('x',encoding='utf-8',newline='\n') as out:out.write(text)
    compile(text,str(path),'exec')
    print({'runtime_created':True,'immutable_parent_policy_init_retained':True,'source_sha256':sha(path)},flush=True)


if __name__=='__main__':main()
