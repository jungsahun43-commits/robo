"""Fixed 40-epoch actual belt polygon training, followed by untouched VAL."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.belt_segmentation import ARCH, create_model, loss, pixel_counts
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.retention_distillation import state_sha256
from prepare_beltcrack_starter import DATA, sha

PROTOCOL = ROOT / 'reports/beltcrack-starter-protocol.json'
RUN = ROOT / 'runs/beltcrack-polygon-starter-v1'
SOURCES = ('safelog_ai/belt_segmentation.py', 'scripts/train_beltcrack_starter.py',
    'scripts/prepare_beltcrack_starter.py', 'scripts/fetch_beltcrack_subset.py',
    'safelog_ai/equipment_anomaly.py', 'safelog_ai/retention_distillation.py',
    'tests/test_belt_segmentation.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def load_images(rows):
    images, masks, valid = [], [], []
    for row in rows:
        values = {}
        for key in ('image', 'mask', 'valid'):
            path = DATA / row[key]
            assert sha(path) == row[key + '_sha256']
            with Image.open(path) as image:
                values[key] = np.array(image)
        images.append(torch.from_numpy(values['image']).permute(2, 0, 1))
        masks.append(torch.from_numpy(values['mask'] > 0).unsqueeze(0))
        valid.append(torch.from_numpy(values['valid'] > 0).unsqueeze(0))
    return torch.stack(images), torch.stack(masks), torch.stack(valid)


def normalize(images):
    x = images.cuda().float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    return (x - mean) / std


def preflight():
    torch.manual_seed(86)
    model = create_model(True).cuda().train()
    previous = state_sha256(model)
    image = torch.randn(2, 3, 640, 640, device='cuda')
    mask = torch.zeros(2, 1, 640, 640, device='cuda')
    mask[:, :, 250:270, 100:350] = 1
    valid = torch.ones_like(mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    value = loss(model(image)['out'], mask, valid, 20)
    value.backward()
    optimizer.step()
    assert torch.isfinite(value) and previous != state_sha256(model)
    model.eval()
    copy = create_model(False).eval()
    copy.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
    with torch.no_grad():
        gpu, cpu = model(image)['out'].cpu(), copy(image.cpu())['out']
    assert torch.allclose(gpu, cpu, atol=0.0001, rtol=0.001)
    new(ROOT / 'runs/beltcrack-starter-preflight.json', {'status': 'passed', 'disposable_updates': 1,
        'actual_epochs': 0, 'strict_cpu_gpu_reload': True, 'source_sha256': {n: sha(ROOT / n) for n in SOURCES}})
    print('Belt GPU preflight passed', flush=True)


def train():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert protocol['manifest_sha256'] == sha(DATA / 'manifest.json')
    assert protocol['epochs'] == 40
    assert not RUN.exists()
    RUN.mkdir()
    manifest = read(DATA / 'manifest.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'train']
    images, masks, valid = load_images(rows)
    positive = sum(r['positive_pixels'] for r in rows)
    known = sum(r['valid_pixels'] for r in rows)
    weight = min(20, max(1, (known - positive) / positive))
    assert weight == protocol['pixel_positive_weight']
    torch.manual_seed(87)
    model = create_model(True).cuda()
    initial = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0.000005)
    history, updates = [], 0
    started = time.monotonic()
    for epoch in range(1, 41):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(rows)).split(8):
            x = normalize(images[batch])
            target, known_mask = masks[batch].cuda(), valid[batch].cuda()
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(x)['out'], target, known_mask, weight)
            assert torch.isfinite(value)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += float(value.detach()) * len(batch)
        scheduler.step()
        entry = {'epoch': epoch, 'train_loss': total / len(rows), 'optimizer_updates': updates,
                 'elapsed_minutes': (time.monotonic() - started) / 60}
        history.append(entry)
        print(json.dumps(entry), flush=True)
    assert initial != state_sha256(model)
    model.eval()
    metadata = {'architecture': ARCH, 'class': 'belt_crack', 'epochs': 40, 'input_size': 640,
        'threshold': 0.5, 'manifest_sha256': sha(DATA / 'manifest.json'), 'protocol_sha256': sha(PROTOCOL),
        'optimizer_updates': updates, 'train_photos': len(rows), 'source_test_used': False,
        'initial_state_sha256': initial, 'final_state_sha256': state_sha256(model),
        'pretrained_source': 'Torchvision LRASPP COCO-with-VOC labels, fresh 1-class heads; no building weights'}
    torch.save({**metadata, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, RUN / 'final.pt')
    copy = create_model(False).eval()
    copy.load_state_dict(torch.load(RUN / 'final.pt', map_location='cpu', weights_only=True)['state_dict'], strict=True)
    with torch.no_grad():
        gpu = model(normalize(images[:2]))['out'].cpu()
        cpu = copy(normalize(images[:2]).cpu())['out']
    assert torch.allclose(gpu, cpu, atol=0.0001, rtol=0.001)
    new(RUN / 'history.json', history)
    new(ROOT / 'reports/beltcrack-starter-training.json', {**metadata, 'weights_sha256': sha(RUN / 'final.pt'),
        'elapsed_training_minutes': (time.monotonic() - started) / 60,
        'strict_cpu_gpu_reload_passed': True, 'validation_used_for_selection': False,
        'source_sha256': protocol['source_sha256']})
    print('Belt final model frozen, no validation inference yet', flush=True)


def evaluate():
    training = read(ROOT / 'reports/beltcrack-starter-training.json')
    assert training['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert training['weights_sha256'] == sha(RUN / 'final.pt')
    checkpoint = torch.load(RUN / 'final.pt', map_location='cpu', weights_only=True)
    model = create_model(False).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    manifest = read(DATA / 'manifest.json')
    assert training['manifest_sha256'] == sha(DATA / 'manifest.json')
    rows = [r for r in manifest['rows'] if r['split'] == 'val']
    images, masks, valid = load_images(rows)
    counts = {name: 0 for name in ('tp', 'fp', 'tn', 'fn')}
    group_counts, scores = {}, []
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            end = start + 8
            probability = model(normalize(images[start:end]))['out'].sigmoid()
            target, known = masks[start:end].cuda(), valid[start:end].cuda()
            local = pixel_counts(probability, target, known)
            for key in counts:
                counts[key] += local[key]
            scores.extend((probability * known).flatten(1).amax(1).cpu().double().tolist())
            for i, row in enumerate(rows[start:end]):
                group = group_counts.setdefault(row['sequence'], {name: 0 for name in counts})
                local = pixel_counts(probability[i:i+1], target[i:i+1], known[i:i+1])
                for key in group:
                    group[key] += local[key]
    tp, fp, tn, fn = (counts[k] for k in ('tp', 'fp', 'tn', 'fn'))
    photo = classification_rates([int(r['photo_positive']) for r in rows], scores, 0.5)
    result = {'status': 'evaluated', 'split': 'publisher val subset, original disjoint sequences',
        'val_photos': len(rows), 'val_sequences': len(group_counts), 'pixel_counts': counts,
        'pixel_iou': tp / (tp+fp+fn), 'pixel_dice': 2*tp / (2*tp+fp+fn),
        'pixel_fnr': fn / (tp+fn), 'pixel_fpr': fp / (tn+fp),
        'photo_rates': photo, 'threshold': 0.5, 'source_test_used': False,
        'independent_factory_error_rate': False, 'source_sequence_correlation_not_removed': True,
        'pixel_truth_at_model_resolution': True, 'padding_excluded': True,
        'default_app_model_changed': False, 'weights_sha256': training['weights_sha256'],
        'photo_fpr_fnr_below_5_percent': photo['fpr'] is not None and photo['fnr'] is not None and photo['fpr'] < 0.05 and photo['fnr'] < 0.05,
        'per_sequence_pixel_counts': group_counts}
    new(RUN / 'val-predictions.json', {'ids': [r['id'] for r in rows], 'photo_scores': scores, 'result': result})
    new(ROOT / 'reports/beltcrack-starter-results.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'train', 'evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'preflight': preflight, 'train': train, 'evaluate': evaluate}[args.mode]()
