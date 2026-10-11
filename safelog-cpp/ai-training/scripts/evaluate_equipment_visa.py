"""One test evaluation of fixed final models and TRAIN-only calibration thresholds."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.equipment_anomaly import (ARCHITECTURE, EquipmentPyramid, appearance_maps,
    classification_rates, rank_auroc)
from train_equipment_visa import DATA, SOURCES, PROTOCOL, tensors, read, write_new
from fetch_equipment_visa import sha


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_path = ROOT / 'reports/equipment-visa-frozen-operating-points.json'
    frozen = read(frozen_path)
    assert frozen['status'] == 'frozen_before_test_inference'
    assert frozen['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert frozen['protocol_sha256'] == sha(PROTOCOL)
    output = ROOT / 'reports/equipment-visa-results.json'
    assert not output.exists(), 'Do not repeatedly tune this test set'
    manifest = read(DATA / 'manifest.json')
    results = []
    for entry in frozen['categories']:
        checkpoint_path = ROOT / entry['checkpoint']
        assert entry['checkpoint_sha256'] == sha(checkpoint_path)
        assert entry['manifest_sha256'] == sha(DATA / 'manifest.json')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['architecture'] == ARCHITECTURE
        assert checkpoint['equipment_category'] == entry['equipment_category']
        assert checkpoint['threshold'] == entry['threshold']
        student, teacher = EquipmentPyramid(False).cuda().eval(), EquipmentPyramid(False).cuda().eval()
        student.load_state_dict(checkpoint['student'], strict=True)
        teacher.load_state_dict(checkpoint['teacher'], strict=True)
        rows = [r for r in manifest['rows'] if r['object'] == entry['equipment_category'] and r['phase'] == 'test']
        assert all(r['split'] == 'test' for r in rows)
        images = tensors(rows)
        scores = []
        with torch.no_grad():
            for image in images.split(32):
                image = image.cuda()
                scores.extend(appearance_maps(student(image), teacher(image)).flatten(1).amax(dim=1).cpu().double().tolist())
        labels = [int(r['label'] == 'anomaly') for r in rows]
        rates = classification_rates(labels, scores, entry['threshold'])
        result = {'category': entry['equipment_category'], **rates,
                  'image_auroc': rank_auroc(labels, scores), 'threshold': entry['threshold'],
                  'checkpoint_sha256': entry['checkpoint_sha256'],
                  'fpr_and_fnr_below_5_percent': rates['fpr'] < 0.05 and rates['fnr'] < 0.05}
        write_new(checkpoint_path.parent / 'test-predictions.json', {'image_ids': [r['image'] for r in rows],
            'labels': labels, 'scores': scores, 'threshold': entry['threshold'], 'results': result,
            'operating_points_sha256': sha(frozen_path)})
        results.append(result)
        print(json.dumps(result), flush=True)
        del student, teacher, images
        torch.cuda.empty_cache()
    write_new(output, {'status': 'evaluated', 'scope': 'VisA official PCB image-level external benchmark test',
        'results': results, 'all_8_fpr_fnr_below_5_percent': all(r['fpr_and_fnr_below_5_percent'] for r in results),
        'thresholds_fixed_before_test': True, 'model_selected_by': 'Fixed final epoch 20, not test or calibration best',
        'source_test_used_for_training_or_calibration': False, 'source_test_inference_passes': 1,
        'normal_calibration_alpha': 0.05, 'alpha_is_not_guaranteed_test_fpr': True,
        'independent_factory_photos': False, 'pixel_segmentation_metrics_computed': False,
        'default_app_model_changed': False, 'source_sha256': frozen['source_sha256'],
        'frozen_operating_points_sha256': sha(frozen_path)})


if __name__ == '__main__':
    main()
