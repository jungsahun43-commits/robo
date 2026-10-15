"""Capacity-controlled follow-up after training-only v26 overfit diagnosis."""
from torch import nn
from .gearbox_supervised import GearboxMain, DIM


class LinearGearboxMain(GearboxMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.gearbox_head = nn.Linear(DIM, 1)
        nn.init.zeros_(self.gearbox_head.weight)
        nn.init.zeros_(self.gearbox_head.bias)
