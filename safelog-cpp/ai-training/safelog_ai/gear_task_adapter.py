"""Task-specific stage3 residual; all original shared backbone modules retained."""
import torch
from .gearbox_linear import LinearGearboxMain
from .gearbox_supervised import gearbox_descriptor
from .gear_adapter_probe import ZeroFineAdapterProbe,tail_from_fine

VERSION='safelog-gear-task-adapter-v31'


def prefix_to_fine(backbone,image):
    value=image
    for name,module in backbone.items():
        value=module(value)
        if name=='3':return value
    raise ValueError('Expected preserved encoder stage3')


def adapted_descriptor(backbone,adapter,fine):
    return gearbox_descriptor(tail_from_fine(backbone,adapter(fine.half().float())))


class GearTaskAdapterMain(LinearGearboxMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.gear_task_adapter=ZeroFineAdapterProbe()

    def forward_gear_adapter_features(self,fine):
        if fine.ndim!=4 or tuple(fine.shape[1:])!=(24,64,64):raise ValueError('Source fine24 at64x64 required')
        return self.forward_gearbox_descriptor(adapted_descriptor(self.backbone,self.gear_task_adapter,fine))

    def forward_gear_adapter_patch(self,image):
        if tuple(image.shape[1:])!=(3,256,256):raise ValueError('RGB256 preprocessed source patch required')
        return self.forward_gear_adapter_features(prefix_to_fine(self.backbone,image))
