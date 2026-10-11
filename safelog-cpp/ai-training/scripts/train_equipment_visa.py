"""Real neural optimization using only normal PCB TRAIN photos, no test selection."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.equipment_anomaly import (ARCHITECTURE, CATEGORIES, EquipmentPyramid,
    matching_loss, appearance_maps, normal_calibration_threshold)
from safelog_ai.retention_distillation import state_sha256
from fetch_equipment_visa import sha

DATA = ROOT / 'data/equipment-visa'
RUN = ROOT / 'runs/equipment-visa-stfpm-v1'
PROTOCOL = ROOT / 'reports/equipment-visa-study-protocol.json'
SOURCES = ('safelog_ai/equipment_anomaly.py', 'scripts/fetch_equipment_visa.py',
           'scripts/prepare_equipment_visa.py', 'scripts/train_equipment_visa.py',
           'scripts/evaluate_equipment_visa.py', 'scripts/infer_equipment_visa.py',
           'tests/test_equipment_anomaly.py', 'safelog_ai/retention_distillation.py',
           'scripts/equipment_visa_study.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def tensors(rows):
    images = []
    for row in rows:
        path = DATA / row['prepared']
        assert sha(path) == row['prepared_sha256']
        with Image.open(path) as image:
            images.append(normalize(to_tensor(image.convert('RGB')),
                                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return torch.stack(images)


def initialize(seed, pretrained_teacher=True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    teacher = EquipmentPyramid(pretrained_teacher).cuda().eval()
    teacher.requires_grad_(False)
    student = EquipmentPyramid(False).cuda()
    return student, teacher


def preflight():
    student, teacher = initialize(80)
    previous = state_sha256(student)
    teacher_hash = state_sha256(teacher)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.003)
    image = torch.randn(2, 3, 256, 256, device='cuda')
    student.train()
    with torch.no_grad():
        targets = teacher(image)
    loss = matching_loss(student(image), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert previous != state_sha256(student)
    assert teacher_hash == state_sha256(teacher)
    assert all(p.grad is None for p in teacher.parameters())
    student.eval()
    with torch.no_grad():
        maps = appearance_maps(student(image), teacher(image))
    assert maps.shape == (2, 256, 256) and torch.isfinite(maps).all()
    copy = EquipmentPyramid(False).eval()
    copy.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()}, strict=True)
    write_new(ROOT / 'runs/equipment-visa-preflight.json', {'status': 'passed',
        'disposable_gpu_updates': 1, 'training_epochs': 0, 'teacher_state_sha256': teacher_hash,
        'teacher_source_weights_sha256': sha(Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'),
        'teacher_frozen': True, 'student_changed': True, 'strict_cpu_reload': True,
        'map_shape': list(maps.shape), 'source_sha256': {n: sha(ROOT / n) for n in SOURCES}})
    print('Equipment GPU preflight passed', flush=True)


def train():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256'] == {n: sha(ROOT / n) for n in SOURCES}
    assert protocol['manifest_sha256'] == sha(DATA / 'manifest.json')
    assert protocol['preflight_sha256'] == sha(ROOT / 'runs/equipment-visa-preflight.json')
    assert protocol['epochs_per_category'] == 20 and protocol['batch_size'] == 32
    assert not RUN.exists(), 'Never overwrite a previous run'
    RUN.mkdir()
    manifest = read(DATA / 'manifest.json')
    assert manifest['normal_only_fit'] and manifest['source_test_in_training'] is False
    started = time.monotonic()
    operating_points = []
    for index, category in enumerate(CATEGORIES):
        rows = [r for r in manifest['rows'] if r['object'] == category and r['phase'] == 'fit']
        calibration = [r for r in manifest['rows'] if r['object'] == category and r['phase'] == 'calibration']
        assert all(r['split'] == 'train' and r['label'] == 'normal' for r in rows + calibration)
        fit = tensors(rows)
        holdout = tensors(calibration)
        student, teacher = initialize(81 + index)
        initial_student = state_sha256(student)
        teacher_hash = state_sha256(teacher)
        assert teacher_hash == protocol['teacher_state_sha256']
        optimizer = torch.optim.Adam(student.parameters(), lr=0.003, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)
        history, steps = [], 0
        for epoch in range(1, 21):
            student.train()
            order = torch.randperm(len(fit))
            total_loss = 0.0
            for batch in order.split(32):
                image = fit[batch].cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    target = teacher(image)
                loss = matching_loss(student(image), target)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
                optimizer.step()
                steps += 1
                total_loss += float(loss.detach()) * len(batch)
            scheduler.step()
            record = {'category': category, 'epoch': epoch, 'normal_train_loss': total_loss / len(fit),
                      'optimizer_steps': steps, 'elapsed_minutes': (time.monotonic() - started) / 60}
            history.append(record)
            print(json.dumps(record), flush=True)
        assert teacher_hash == state_sha256(teacher), 'Teacher changed'
        assert all(p.grad is None for p in teacher.parameters())
        assert initial_student != state_sha256(student), 'No neural training occurred'
        student.eval()
        scores = []
        with torch.no_grad():
            for image in holdout.split(32):
                image = image.cuda()
                score = appearance_maps(student(image), teacher(image)).flatten(1).amax(dim=1)
                scores.extend(score.cpu().double().tolist())
        threshold = normal_calibration_threshold(scores)
        directory = RUN / category
        directory.mkdir()
        checkpoint = directory / 'final.pt'
        metadata = {'architecture': ARCHITECTURE, 'equipment_category': category, 'image_size': 256,
                    'epochs': 20, 'manifest_sha256': sha(DATA / 'manifest.json'),
                    'protocol_sha256': sha(PROTOCOL), 'threshold': threshold,
                    'teacher_sha256': teacher_hash, 'normal_fit_photos': len(rows),
                    'normal_calibration_photos': len(calibration), 'optimizer_steps': steps,
                    'source_test_used': False, 'score_is_probability': False}
        torch.save({**metadata, 'student': {k: v.cpu() for k, v in student.state_dict().items()},
                    'teacher': {k: v.cpu() for k, v in teacher.state_dict().items()}}, checkpoint)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        cpu_student, cpu_teacher = EquipmentPyramid(False).eval(), EquipmentPyramid(False).eval()
        cpu_student.load_state_dict(restored['student'], strict=True)
        cpu_teacher.load_state_dict(restored['teacher'], strict=True)
        with torch.no_grad():
            gpu = appearance_maps(student(holdout[:2].cuda()), teacher(holdout[:2].cuda())).cpu()
            cpu = appearance_maps(cpu_student(holdout[:2]), cpu_teacher(holdout[:2]))
        assert torch.allclose(gpu, cpu, atol=1e-5, rtol=1e-3), 'CPU/GPU inference mismatch'
        write_new(directory / 'history.json', history)
        write_new(directory / 'calibration.json', {'normal_image_ids': [r['image'] for r in calibration],
            'scores': scores, 'threshold': threshold, 'threshold_rule': 'ceil((n+1)*0.95) ordered normal score; strict >',
            'test_used_for_threshold': False})
        operating_points.append({**metadata, 'checkpoint': str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
            'checkpoint_sha256': sha(checkpoint), 'calibration_sha256': sha(directory / 'calibration.json'),
            'strict_cpu_gpu_reload_passed': True})
        write_new(directory / 'training.json', operating_points[-1])
        del student, teacher, optimizer, fit, holdout, cpu_student, cpu_teacher
        torch.cuda.empty_cache()
    write_new(ROOT / 'reports/equipment-visa-frozen-operating-points.json', {
        'status': 'frozen_before_test_inference', 'categories': operating_points,
        'completed_category_model_epochs': 80, 'source_test_used': False,
        'source_sha256': protocol['source_sha256'], 'protocol_sha256': sha(PROTOCOL),
        'elapsed_training_minutes': (time.monotonic() - started) / 60})
    print('Four PCB models and thresholds frozen; no test inference yet', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'train'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    import os
    os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch-equipment')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preflight() if args.mode == 'preflight' else train()


if __name__ == '__main__':
    main()
