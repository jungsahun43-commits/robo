"""Deterministic, bounded CPU caches with unchanged facility and belt truth grids."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import normalize, to_tensor

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.presence_classifier import image_transform
DATA = ROOT/'data/unified-main-v1'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_batch(images, device='cuda'):
    values = images.to(device).float()/255
    mean = values.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std = values.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (values-mean)/std


def photo(row):
    with Image.open(ROOT/row['image']) as original:
        rgb = original.convert('RGB')
        if row['task'] == 'facility_surface': rgb = rgb.resize((640,640),Image.Resampling.BICUBIC)
        elif row['task'] == 'conveyor_foreign':
            scale = 640/max(rgb.size); w,h = [max(1,round(v*scale)) for v in rgb.size]
            image = Image.new('RGB',(640,640),(123,116,103))
            image.paste(rgb.resize((w,h),Image.Resampling.BILINEAR),((640-w)//2,(640-h)//2)); rgb = image
        assert rgb.size == (640,640)
        return torch.from_numpy(np.array(rgb)).permute(2,0,1)


def targets(row):
    task = row['task']
    if task == 'conveyor_foreign': return {'label':torch.tensor(row['label'],dtype=torch.float32)}
    if task == 'belt_damage':
        result = {}
        for k in ['mask','valid']:
            with Image.open(ROOT/row[k]) as image: result[k] = torch.from_numpy(np.array(image)>0).unsqueeze(0)
        return result
    with np.load(ROOT/row['pixel_target'],allow_pickle=False) as target:
        mask = torch.from_numpy(target['mask'].copy()); pixel_known = torch.from_numpy(target['known'].copy())
    labels = torch.tensor(row['targets'],dtype=torch.float32); aux = torch.tensor(row['auxiliary_targets'],dtype=torch.float32)
    return {'labels':labels.clamp_min(0),'known':labels>=0,'mask':mask,'pixel_known':pixel_known,
            'aux':aux.clamp_min(0),'aux_known':aux>=0}


class Cache:
    def __init__(self, rows, indices, limit=256):
        from collections import OrderedDict
        self.rows,self.allowed,self.limit = rows,set(indices),limit
        self.values = OrderedDict()
    def row(self, index):
        if index not in self.allowed: raise ValueError('Unscheduled or held-out record cannot enter TRAIN cache')
        if index not in self.values:
            row = self.rows[index]; self.values[index] = (photo(row),targets(row))
            if len(self.values)>self.limit: self.values.popitem(last=False)
        self.values.move_to_end(index); return self.values[index]
    def batch(self, indices, device='cuda', training=False):
        values = [self.row(int(i)) for i in indices]
        images = torch.stack([v[0] for v in values]); truth = {k:torch.stack([v[1][k] for v in values]).to(device).float() for k in values[0][1]}
        if training:
            flips = torch.rand(len(images))<.5; images[flips] = images[flips].flip(-1)
            if 'mask' in truth: truth['mask'][flips.to(device)] = truth['mask'][flips.to(device)].flip(-1)
            if 'valid' in truth: truth['valid'][flips.to(device)] = truth['valid'][flips.to(device)].flip(-1)
        return normalize_batch(images,device),truth
