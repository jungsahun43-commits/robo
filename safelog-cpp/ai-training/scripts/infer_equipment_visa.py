"""Isolated research inference CLI; require explicit PCB category and checkpoint."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.equipment_anomaly import (ARCHITECTURE, EquipmentPyramid, appearance_maps,
    inspection_result, CATEGORIES)
from fetch_equipment_visa import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--heatmap', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE or checkpoint['equipment_category'] != args.category:
        raise ValueError('Checkpoint equipment category mismatch')
    student, teacher = EquipmentPyramid(False).eval(), EquipmentPyramid(False).eval()
    student.load_state_dict(checkpoint['student'], strict=True)
    teacher.load_state_dict(checkpoint['teacher'], strict=True)
    with Image.open(args.image) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    image = normalize(to_tensor(rgb), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    with torch.no_grad():
        heatmap = appearance_maps(student(image), teacher(image))[0]
    result = inspection_result(args.category, float(heatmap.max()), checkpoint['threshold'], sha(args.checkpoint))
    if args.heatmap:
        import numpy as np
        assert not args.heatmap.exists(), 'Preserve existing heatmap'
        Image.fromarray(np.clip(heatmap.numpy() / max(float(heatmap.max()), 1e-12) * 255, 0, 255).astype('uint8')).save(args.heatmap)
        result['heatmap'] = str(args.heatmap.resolve())
        result['heatmap_scale'] = 'Per-photo relative intensity for visualization; not a calibrated pixel verdict'
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
