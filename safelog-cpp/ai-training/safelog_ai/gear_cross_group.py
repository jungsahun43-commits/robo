"""One shared task-adapter forward for patch BCE and cross-part regularization."""
import numpy as np
import torch
from .gear_task_adapter import GearTaskAdapterMain, adapted_descriptor, prefix_to_fine

VERSION = 'safelog-gear-cross-group-v33'


def sample_cross_group_batch(rows, rng):
    """Four groups per actual label, two distinct patches in each chosen group."""
    indices = []
    for label in (0, 1):
        pools = {}
        for i, row in enumerate(rows):
            if row['label'] == label:
                pools.setdefault(str(row['group_id']), []).append(i)
        eligible = sorted(group for group, ids in pools.items() if len(ids) >= 2)
        if len(eligible) < 4:
            raise ValueError('Four actual FIT groups with two patches per label required')
        for group in rng.choice(eligible, 4, replace=False):
            indices.extend(rng.choice(pools[group], 2, replace=False).tolist())
    return np.asarray(indices, dtype=np.int64)


def cross_group_forward(backbone, adapter, head, mean, std, fine):
    descriptor = adapted_descriptor(backbone, adapter, fine)
    embedding = ((descriptor.float() - mean) / std.clamp_min(1e-3)).clamp(-8, 8)
    return head(embedding).flatten(), embedding


class GearCrossGroupMain(GearTaskAdapterMain):
    def forward_gear_cross_group_features(self, fine):
        if fine.ndim != 4 or tuple(fine.shape[1:]) != (24, 64, 64):
            raise ValueError('Source fine24 at64x64 required')
        return cross_group_forward(self.backbone, self.gear_task_adapter,
                                   self.gearbox_head, self.gearbox_mean,
                                   self.gearbox_std, fine)

    def forward_gear_cross_group_patch(self, image):
        if tuple(image.shape[1:]) != (3, 256, 256):
            raise ValueError('Preprocessed source RGB256 patch required')
        return self.forward_gear_cross_group_features(prefix_to_fine(self.backbone, image))
