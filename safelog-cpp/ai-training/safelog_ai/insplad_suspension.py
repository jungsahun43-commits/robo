"""A new explicit suspension-rust branch with both approved branches preserved."""
from .insplad_glass import InspladGlassMain
from .insplad_classifier import CorrosionClassifier
from .fine_feature_probe import forward_with_fine
from .gearbox_supervised import gearbox_descriptor


class InspladSuspensionMain(InspladGlassMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.insplad_suspension_classifier=CorrosionClassifier()

    def forward_insplad_suspension_descriptor(self,descriptor):
        return self.insplad_suspension_classifier(descriptor)

    def forward_insplad_suspension(self,image):
        return self.forward_insplad_suspension_descriptor(gearbox_descriptor(forward_with_fine(self.backbone,image)))
