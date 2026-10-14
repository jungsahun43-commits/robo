"""Extend the accepted single-backbone photo memory to explicit component types."""
import torch
from .equipment_expansion import CATEGORIES
from .equipment_reference_photo import ReferencePhotoMain, REFERENCE_CATEGORIES as PRIOR_CATEGORIES

ARCH = 'safelog-equipment-component-reference-v3'
NEW_CATEGORIES = ('ad_screw', 'loco_splicing_connectors', 'mpdd_connector')


class ComponentReferenceMain(ReferencePhotoMain):
    def __init__(self, reference_counts):
        if any(c not in CATEGORIES or type(n) is not int or n < 1 for c, n in reference_counts.items()):
            raise ValueError('Positive reference patch counts for supported explicit categories required')
        super().__init__({c: n for c, n in reference_counts.items() if c in PRIOR_CATEGORIES})
        for category, count in reference_counts.items():
            if category not in PRIOR_CATEGORIES:
                self.register_buffer('normal_reference_'+category, torch.zeros(count, 1000, dtype=torch.float16))
        self.reference_counts = dict(reference_counts)
