"""Freeze full TRAIN photo predictions for error review, never evaluation gold edits."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.presence_classifier import build_model
from scripts.train_facility_target import read, save, sha, dacl_items, split_supplemental, predict
from scripts.train_facility_spatial import MiningPhotos


def verified_full_train(manifest, classes):
    if manifest['split'] != 'train' or manifest['classes'] != classes:
        raise ValueError('Review input must be the matching TRAIN manifest')
    original, source_classes = dacl_items('train', 640)
    split = split_supplemental()
    code = read(ROOT / 'data/codebrim-training/train.json')
    if source_classes != classes or split['classes'] != classes or code['classes'] != classes or code['split'] != 'train':
        raise ValueError('Source TRAIN classes or split changed')
    base = original + [{**i, 'domain': 'damsegment'} for i in split['train']] + code['items']
    if any(i.get('split') != 'train' or i.get('domain') != 'codebrim' for i in code['items']):
        raise ValueError('CODEBRIM review rows must be asserted TRAIN')
    rows = manifest['items'][:manifest['full_count']]
    if len(rows) != len(base) or any(a['image'] != b['image'] or a['targets'] != b['targets'] or a['domain'] != b['domain'] for a, b in zip(rows, base)):
        raise ValueError('Review full rows must exactly match recorded TRAIN images and labels')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='facility-presence-target-auxiliary')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError('Invalid run name')
    run = ROOT / 'runs' / args.name
    weights = run / 'best.pt'
    training = read(run / 'TRAINING.json')
    manifest_path = ROOT / 'data/facility-spatial-training/train.json'
    if training['status'] != 'complete' or sha(weights) != training['weights_sha256']:
        raise ValueError('Review requires completed, frozen weights')
    if sha(manifest_path) != training['spatial_manifest_sha256']:
        raise ValueError('Original spatial TRAIN manifest changed')
    manifest = read(manifest_path)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != training['classes'] or checkpoint['split_sha256'] != sha(run / 'SPLIT.json'):
        raise ValueError('Checkpoint classes or source split changed')
    items = verified_full_train(manifest, checkpoint['classes'])
    output = args.output.resolve() if args.output else run / 'TRAIN-REVIEW-PREDICTIONS.json'
    if output.exists():
        raise ValueError('Preserve existing review predictions; choose a new output')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; explicitly select --device cpu')
    torch.set_num_threads(4)
    model = build_model(len(checkpoint['classes']), architecture=checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device).eval()
    loader = DataLoader(MiningPhotos(items), batch_size=16, num_workers=4, pin_memory=args.device == 'cuda')
    print(f'Scoring {len(items)} full TRAIN records only; no training or gold edits', flush=True)
    targets, probabilities = predict(model, loader, args.device)
    if not np.array_equal(targets, [i['targets'] for i in items]) or not np.isfinite(probabilities).all():
        raise ValueError('Review prediction order or numeric validity failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, {
        'run': args.name, 'split': 'train', 'classes': checkpoint['classes'],
        'weights_sha256': sha(weights), 'manifest_sha256': sha(manifest_path),
        'script_sha256': sha(Path(__file__)), 'split_sha256': sha(run / 'SPLIT.json'),
        'images': [i['image'] for i in items], 'targets': targets.tolist(),
        'probabilities': probabilities.tolist(),
        'scope': 'Full TRAIN photos/patches only, deterministic frozen inference; no validation/test predictions, no label edits, not a generalization accuracy estimate'
    })
    print(f'Saved TRAIN review predictions: {output}', flush=True)


if __name__ == '__main__':
    main()
