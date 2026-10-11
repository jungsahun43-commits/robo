"""CPU research inference for a conveyor photo, with no mechanical verdict."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.belt_segmentation import ARCH, create_model
from prepare_beltcrack_starter import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/beltcrack-polygon-starter-v1/final.pt')
    parser.add_argument('--heatmap', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert checkpoint['architecture'] == ARCH and checkpoint['class'] == 'belt_crack'
    assert checkpoint['input_size'] == 640 and checkpoint['threshold'] == 0.5
    model = create_model(False).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    with Image.open(args.image) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        width, height = rgb.size
        scale = 640 / max(width, height)
        w, h = round(width * scale), round(height * scale)
        left, top = (640 - w) // 2, (640 - h) // 2
        padded = Image.new('RGB', (640, 640), (123, 116, 103))
        padded.paste(rgb.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    image = normalize(to_tensor(padded), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    with torch.no_grad():
        probability = model(image)['out'][0, 0].sigmoid()[top:top+h, left:left+w]
    result = {'schema': 'safelog.equipment.belt-visual.v1', 'equipment_category': 'conveyor_belt',
        'class': 'belt_crack', 'candidate_detected': bool((probability > 0.5).any()),
        'maximum_pixel_score': float(probability.max()), 'pixel_threshold': 0.5,
        'score_is_mechanical_failure_probability': False, 'normal_belt_specificity_validated': False,
        'mechanical_failure_state': 'not_assessed', 'human_confirmation_required': True,
        'scope': 'First belt surface research candidate; no roller/motor/alignment diagnosis',
        'checkpoint_sha256': sha(args.checkpoint), 'input_sha256': sha(args.image),
        'input_size': [width, height]}
    if args.heatmap:
        assert not args.heatmap.exists(), 'Preserve existing output'
        heatmap = Image.fromarray(np.clip(probability.numpy()*255, 0, 255).astype('uint8'))
        heatmap.resize((width, height), Image.Resampling.BILINEAR).save(args.heatmap)
        result['heatmap'] = str(args.heatmap.resolve())
        result['heatmap_scale'] = 'Raw sigmoid pixel scores, 0~255; no per-photo stretching'
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
