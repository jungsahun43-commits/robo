"""One PCB student shared by two sources; category-specific normal CAL thresholds."""
import torch
from .pcb_reverse import ReverseStudent, create_teacher, loss, anomaly_maps, SIZE

ARCH = 'safelog-pcb-pooled-visa-robustad-v1'
CATEGORIES = ('pcb1', 'pcb2', 'pcb3', 'pcb4', 'robustad_pcb')
SOURCES = ('VisA', 'RobustAD')


def balanced_indices(rows, seed, draws_per_category=800):
    if not rows or any(r['phase'] != 'fit' or r['split'] != 'train' or r['label'] != 0 for r in rows):
        raise ValueError('Only author-labelled normal FIT/TRAIN photos may enter the optimizer')
    if set(r['source'] for r in rows) != set(SOURCES):
        raise ValueError('Both distinct publisher sources required')
    if set(r['category'] for r in rows) != set(CATEGORIES) or draws_per_category < 1:
        raise ValueError('All five known PCB types and positive draw count required')
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for category in CATEGORIES:
        pool = torch.tensor([i for i, r in enumerate(rows) if r['category'] == category])
        indices.append(pool[torch.randint(len(pool), (draws_per_category,), generator=generator)])
    combined = torch.cat(indices)
    return combined[torch.randperm(len(combined), generator=generator)]
