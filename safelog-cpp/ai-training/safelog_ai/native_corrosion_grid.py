"""Input-size-only native512 grid; never choose inference crops using annotations."""
import numpy as np


def starts(length):
    if not isinstance(length,int) or length<=0:raise ValueError('Positive image axis required')
    if length<=512:return [0]
    result=list(range(0,length-512+1,384))
    if result[-1]!=length-512:result.append(length-512)
    return result


def grid(size):
    w,h=size
    return [{'left':x,'top':y,'width':min(512,w-x),'height':min(512,h-y)} for y in starts(h) for x in starts(w)]


def annotation_tile(mask,box):
    h,w=mask.shape;x,y,rh,rw=box['left'],box['top'],box['height'],box['width']
    if min(x,y)<0 or not 0<rw<=512 or not 0<rh<=512 or x+rw>w or y+rh>h:raise ValueError('Input grid outside original coordinates')
    if not set(np.unique(mask))<={0,1,255}:raise ValueError('Sparse binary labels required')
    tile=np.full((512,512),255,dtype=np.uint8);tile[:rh,:rw]=mask[y:y+rh,x:x+rw]
    return tile


def fine_cells(target,label):
    if target.shape!=(512,512):raise ValueError('Full512 target required')
    # Geometric label support in stride4 bins, not an encoder receptive-field or
    # actual feature-response/accuracy measurement.
    return int((target==label).reshape(128,4,128,4).any((1,3)).sum())
