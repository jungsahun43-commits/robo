"""Reload the completed experimental weights; verify shapes, not accuracy."""
from pathlib import Path
import json
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH
from scripts.report_facility_building_supplement import CLASSES, CONTROL, TREATMENT
from scripts.train_facility_target import read, save, sha
from scripts.facility_photo_supplement import validate_photo_supplement


def main():
    torch.set_num_threads(4)
    entries = []
    for name in (CONTROL, TREATMENT):
        run = ROOT / 'runs' / name
        training = read(run / 'TRAINING.json')
        weights = run / 'best.pt'
        if training.get('status') != 'complete' or training['weights_sha256'] != sha(weights):
            raise ValueError('Only completed unchanged experimental checkpoints may be verified')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if (checkpoint['architecture'] != ARCH or checkpoint['classes'] != CLASSES
                or checkpoint['auxiliary_classes'] != list(AUX_CLASSES)):
            raise ValueError('Experimental checkpoint public/auxiliary class contract changed')
        model = AuxiliaryClassifier(len(CLASSES), pretrained=False).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        with torch.inference_mode():
            inputs = torch.zeros(1, 3, 640, 640)
            public = model(inputs)
            scores, maps, auxiliary = model.forward_training(inputs)
        if (public.shape != (1, 7) or maps.shape != (1, 7, 80, 80) or auxiliary.shape != (1, 19)
                or not torch.equal(public, scores)
                or not all(torch.isfinite(value).all() for value in (public, maps, auxiliary))):
            raise ValueError('Finite seven-output checkpoint contract failed')
        entries.append({'run': name, 'weights_sha256': sha(weights), 'public_shape': list(public.shape),
                        'private_spatial_shape': list(maps.shape), 'training_auxiliary_shape': list(auxiliary.shape),
                        'public_and_training_public_logits_equal': True, 'all_outputs_finite': True})
    supplement = read(ROOT / 'data/convid-training/train.json')
    core = read(ROOT / 'data/facility-spatial-training/train.json')
    rows = validate_photo_supplement(supplement, CLASSES, ROOT, {i['image'] for i in core['items']})
    profile = ROOT / 'reports/facility-inference-profile.json'
    previous = ROOT / 'reports/facility-inference-profile-round1.json'
    if sha(profile) != sha(previous):
        raise ValueError('App default profile changed during the research experiment')
    interrupted_path = ROOT / 'runs/facility-presence-target-building-control-preflight-aborted/TRAINING.json'
    interrupted = read(interrupted_path) if interrupted_path.exists() else None
    if interrupted and (interrupted.get('status') != 'aborted' or interrupted.get('actual_epochs') != 0):
        raise ValueError('Interrupted setup run must remain separate from completed epochs')
    result = {'status': 'passed', 'classes': CLASSES, 'experiments': entries,
              'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
              'input': 'One all-zero 1x3x640x640 tensor; tensor/API verification only',
              'accuracy_measured_by_this_check': False, 'structural_safety_verified': False,
              'convid_preflight_photos': len(rows), 'convid_manifest_sha256': sha(ROOT / 'data/convid-training/train.json'),
              'shared_trainer_sha256': sha(ROOT / 'scripts/train_facility_spatial.py'),
              'source_validator_sha256': sha(ROOT / 'scripts/facility_photo_supplement.py'),
              'app_profile_sha256': sha(profile), 'app_profile_unchanged': True, 'experimental_models_deployed': False,
              'interrupted_setup': {'completed_epochs': 0, 'last_observed_draws': interrupted['last_observed_sampling_progress'],
                                    'adopted_checkpoint': False} if interrupted else None,
              'verification_script_sha256': sha(Path(__file__))}
    save(ROOT / 'reports/facility-building-technical-verification.json', result)
    print(json.dumps({'status': 'passed', 'checkpoints': len(entries), 'public_outputs': 7,
                      'convid_preflight_photos': len(rows), 'accuracy_measured': False}))


if __name__ == '__main__':
    main()
