"""Read an existing encoder stage without editing its normal forward or weights."""
from collections import OrderedDict


def forward_with_fine(backbone,image):
    if set(backbone.return_layers.values())!={'low','high'}:raise ValueError('Expected preserved two-output backbone')
    result=OrderedDict();value=image;fine=None
    for name,module in backbone.items():
        value=module(value)
        if name=='3':fine=value
        if name in backbone.return_layers:result[backbone.return_layers[name]]=value
    if fine is None or fine.shape[1]!=24 or tuple(fine.shape[-2:])!=tuple(d//4 for d in image.shape[-2:]):
        raise ValueError('Expected existing stage3 fine24 stride4 output')
    result['fine']=fine
    return result
