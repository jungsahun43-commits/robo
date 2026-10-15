"""Reuse the original later modules after stage3; old forward remains untouched."""
from collections import OrderedDict
import torch
from torch import nn


class ZeroFineAdapterProbe(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(24,4,1),nn.GELU(),nn.Conv2d(4,24,1))
        nn.init.zeros_(self[-1].weight);nn.init.zeros_(self[-1].bias)

    def forward(self,fine):return fine+super().forward(fine)


def tail_from_fine(backbone,fine):
    if fine.ndim!=4 or fine.shape[1]!=24 or set(backbone.return_layers.values())!={'low','high'}:raise ValueError('Preserved fine24/low/high contract required')
    result=OrderedDict();seen=False;value=fine
    for name,module in backbone.items():
        if seen:
            value=module(value)
            if name in backbone.return_layers:result[backbone.return_layers[name]]=value
        elif name=='3':seen=True
    if not seen or set(result)!={'low','high'} or result['low'].shape[1]!=40 or result['high'].shape[1]!=960:raise ValueError('Original stage3 then low40/high960 required')
    result['fine']=fine
    return result
