"""Small reverse-distillation-inspired PCB model; not an exact paper reproduction."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur

from .equipment_anomaly import EquipmentPyramid

ARCH = 'safelog-pcb-reverse-resnet18-v1'
CATEGORIES = ('pcb1','pcb2','pcb3','pcb4')
SIZE = 256


def block(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers = nn.Sequential(block(channels,channels),nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))

    def forward(self,x):
        return F.relu(x+self.layers(x))


class ReverseStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(block(448,128),block(128,256))
        self.low = nn.Sequential(Residual(256),Residual(256))
        self.mid = nn.Sequential(block(256,128),Residual(128),Residual(128))
        self.high = nn.Sequential(block(128,64),Residual(64),Residual(64))

    def forward(self,features):
        a,b,c = [value.detach() for value in features]
        merged = torch.cat([F.adaptive_avg_pool2d(a,c.shape[-2:]),F.adaptive_avg_pool2d(b,c.shape[-2:]),c],dim=1)
        low = self.low(self.bottleneck(merged))
        mid = self.mid(F.interpolate(low,size=b.shape[-2:],mode='bilinear',align_corners=False))
        high = self.high(F.interpolate(mid,size=a.shape[-2:],mode='bilinear',align_corners=False))
        return high,mid,low


def create_teacher(pretrained=False):
    encoder = EquipmentPyramid(pretrained).eval()
    encoder.requires_grad_(False)
    return encoder


def loss(predicted,target):
    if len(predicted)!=3 or len(target)!=3 or any(a.shape!=b.shape for a,b in zip(predicted,target)):
        raise ValueError('Three equally shaped feature pairs required')
    return sum((1-F.cosine_similarity(a.flatten(1).float(),b.detach().flatten(1).float(),dim=1)).mean()
               for a,b in zip(predicted,target))


def anomaly_maps(predicted,target):
    maps = [F.interpolate((1-F.cosine_similarity(a.float(),b.float(),dim=1)).unsqueeze(1),
                          size=(SIZE,SIZE),mode='bilinear',align_corners=False)
            for a,b in zip(predicted,target)]
    return gaussian_blur(torch.stack(maps).sum(0),[25,25],[4.,4.]).squeeze(1).clamp_min(0)
