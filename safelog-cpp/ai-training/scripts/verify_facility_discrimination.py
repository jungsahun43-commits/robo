"""Reload the completed ranking pair and check its unchanged inference contract."""
from pathlib import Path
import json
import platform
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier,AUX_CLASSES,ARCH


def main():
    torch.set_num_threads(4)
    protocol_path=ROOT/'reports/facility-target-discrimination-protocol.json'
    protocol=read(protocol_path)
    source_paths={'training_script_sha256':ROOT/'scripts/train_facility_spatial.py',
                  'model_source_sha256':ROOT/'safelog_ai/spatial_classifier.py',
                  'auxiliary_model_source_sha256':ROOT/'safelog_ai/auxiliary_classifier.py'}
    entries=[]
    for name in (protocol['control'],protocol['treatment']):
        run=ROOT/'runs'/name;training=read(run/'TRAINING.json');weights=run/'best.pt'
        if (training['status']!='complete' or training['actual_epochs']!=6
                or training['weights_sha256']!=sha(weights)
                or training.get('study_protocol_sha256')!=sha(protocol_path)):
            raise ValueError('Incomplete, altered or undeclared paired training')
        for key,path in source_paths.items():
            if training[key]!=sha(path):raise ValueError(f'Executed paired source changed: {key}')
        if training['target_ranking']['helper_sha256']!=sha(ROOT/'scripts/facility_target_ranking.py'):
            raise ValueError('Executed ranking helper changed')
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        if (checkpoint['architecture']!=ARCH or checkpoint['classes']!=protocol['classes']
                or checkpoint['auxiliary_classes']!=list(AUX_CLASSES)
                or checkpoint['selection_split']!='val'):
            raise ValueError('Public/auxiliary output or source selection changed')
        model=AuxiliaryClassifier(7,pretrained=False).eval()
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        with torch.inference_mode():
            inputs=torch.zeros(1,3,640,640)
            public=model(inputs);scores,maps,auxiliary=model.forward_training(inputs)
        if (public.shape!=(1,7) or maps.shape!=(1,7,80,80) or auxiliary.shape!=(1,19)
                or not torch.equal(public,scores)
                or not all(torch.isfinite(value).all() for value in (public,maps,auxiliary))):
            raise ValueError('Finite unchanged inference contract failed')
        entries.append({'run':name,'weights_sha256':sha(weights),
                        'public_shape':list(public.shape),'private_spatial_shape':list(maps.shape),
                        'auxiliary_shape':list(auxiliary.shape),'public_training_logits_equal':True,
                        'outputs_finite':True,'study_protocol_sha256':training['study_protocol_sha256']})
    profile=ROOT/'reports/facility-inference-profile.json'
    if sha(profile)!=sha(ROOT/'reports/facility-inference-profile-round1.json'):
        raise ValueError('App profile changed during research')
    preflight=read(ROOT/'runs/facility-discrimination-preflight.json')
    if preflight['status']!='passed' or not preflight['finite_loss_and_gradients']:
        raise ValueError('Actual TRAIN GPU integration preflight missing')
    freeze=read(ROOT/'runs/facility-discrimination-source-freeze.json')
    for relative,proof in freeze['sources'].items():
        current=sha(ROOT/relative)
        if not proof['equal'] or proof['git_blob_sha256']!=current or proof['executed_file_sha256']!=current:
            raise ValueError('Pre-training committed source snapshot differs')
    tests=read(ROOT/'runs/facility-discrimination-test-verification.json')
    if tests['status']!='passed' or tests['failures']!=0 or tests['errors']!=0:
        raise ValueError('Recorded full-suite verification failed')
    result={'status':'passed','experiments':entries,'python':platform.python_version(),'torch':torch.__version__,
            'device':'cpu','check_input':'One zero 1x3x640x640 tensor per completed checkpoint; shape check only',
            'actual_train_gpu_preflight':preflight,'actual_train_gpu_preflight_sha256':sha(ROOT/'runs/facility-discrimination-preflight.json'),
            'pre_training_source_snapshot':freeze,
            'actual_test_verification':tests,
            'executed_sources_sha256':{key:sha(path) for key,path in source_paths.items()},
            'ranking_helper_sha256':sha(ROOT/'scripts/facility_target_ranking.py'),
            'study_protocol_sha256':sha(protocol_path),'app_profile_sha256':sha(profile),
            'app_profile_unchanged':True,'research_models_deployed':False,
            'accuracy_measured_by_this_check':False,'independent_field_safety_verified':False,
            'verification_script_sha256':sha(Path(__file__))}
    save(ROOT/'reports/facility-target-discrimination-technical-verification.json',result)
    print(json.dumps({'status':'passed','checkpoints':len(entries),'public_outputs':7,'accuracy_measured':False}))


if __name__=='__main__':main()
