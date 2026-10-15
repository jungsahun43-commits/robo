"""Glass-insulator source classification, retaining the approved corrosion branch."""
import torch
from PIL import Image
from .insplad_classifier import InspladMain,CorrosionClassifier
from .fine_feature_probe import forward_with_fine

SIDE=384


def glass_geometry(size):
    width,height=map(int,size)
    if min(width,height)<=0:raise ValueError('Positive input dimensions required')
    factor=min(SIDE/width,SIDE/height)
    resized=(max(1,round(width*factor)),max(1,round(height*factor)))
    offset=((SIDE-resized[0])//2,(SIDE-resized[1])//2)
    return [offset[0],offset[1],offset[0]+resized[0],offset[1]+resized[1]]


def glass_view(rgb):
    box=glass_geometry(rgb.size)
    view=Image.new('RGB',(SIDE,SIDE),(127,127,127))
    view.paste(rgb.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.BICUBIC),(box[0],box[1]))
    return view,box


def glass_descriptor(features,boxes):
    """Pool centers inside the unpadded input rectangle, never a defect mask."""
    result=[]
    for name,channels in (('fine',24),('low',40),('high',960)):
        value=features[name].half().float()
        if value.ndim!=4 or value.shape[1]!=channels or len(boxes)!=len(value):
            raise ValueError('Preserved encoder geometry and one input box per photo required')
        height,width=value.shape[-2:]
        xx=(torch.arange(width,device=value.device)+.5)*SIDE/width
        yy=(torch.arange(height,device=value.device)+.5)*SIDE/height
        pooled=[]
        for row,box in zip(value,boxes):
            x0,y0,x1,y1=map(int,box)
            if not (0<=x0<x1<=SIDE and 0<=y0<y1<=SIDE):raise ValueError('Valid letterbox rectangle required')
            inside=(yy[:,None]>=y0)&(yy[:,None]<y1)&(xx[None,:]>=x0)&(xx[None,:]<x1)
            if not inside.any():raise ValueError('Input too narrow at encoder grid; manual review required')
            cells=row[:,inside]
            pooled.append(torch.stack((cells.mean(1),cells.std(1,correction=0),cells.amax(1)),0))
        values=torch.stack(pooled)
        result.extend((values[:,0],values[:,1],values[:,2]))
    return torch.cat(result,1).half().float()


def assert_preserved(parent,candidate):
    for name,value in parent['state_dict'].items():
        if name not in candidate['state_dict'] or not torch.equal(value,candidate['state_dict'][name]):
            raise ValueError('Protected research tensor changed: '+name)
    for name,value in parent.items():
        if name!='state_dict' and (name not in candidate or value!=candidate[name]):
            raise ValueError('Protected research metadata changed: '+name)


class InspladGlassMain(InspladMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        # Fresh parameters with the same generic MLP architecture; no rust-head copy.
        self.insplad_glass_classifier=CorrosionClassifier()

    def forward_insplad_glass_descriptor(self,descriptor):
        return self.insplad_glass_classifier(descriptor)

    def forward_insplad_glass(self,image,boxes):
        return self.forward_insplad_glass_descriptor(glass_descriptor(forward_with_fine(self.backbone,image),boxes))
