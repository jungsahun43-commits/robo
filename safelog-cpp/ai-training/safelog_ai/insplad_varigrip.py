"""Three original vari-grip photo conditions, using the existing shared encoder."""
import torch
from torch import nn
from .insplad_suspension import InspladSuspensionMain
from .fine_feature_probe import forward_with_fine
from .gearbox_supervised import gearbox_descriptor

CONDITIONS=('good','bird-nest','rust')


class VariGripClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(3072))
        self.register_buffer('std',torch.ones(3072))
        self.head=nn.Sequential(nn.Linear(3072,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,descriptor):return self.head(((descriptor-self.mean)/self.std).clamp(-8,8))


class InspladVariGripMain(InspladSuspensionMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.insplad_varigrip_classifier=VariGripClassifier()

    def forward_insplad_varigrip_descriptor(self,descriptor):return self.insplad_varigrip_classifier(descriptor)

    def forward_insplad_varigrip(self,image):
        return self.forward_insplad_varigrip_descriptor(gearbox_descriptor(forward_with_fine(self.backbone,image)))


def condition_decisions(probabilities,cut):
    """Threshold only the overall non-good score, then name the leading fault."""
    score=1-probabilities[:,0]
    subtype=probabilities[:,1:].argmax(1)+1
    return score,torch.where(score>cut,subtype,torch.zeros_like(subtype))
