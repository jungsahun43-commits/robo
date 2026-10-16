"""Provided subpixel rectangles interpreted through ImageJ's documented pixel bounds."""
import math,struct
import numpy as np


def subpixel_rectangle(raw):
    if not 64<=len(raw)<=1024**2 or raw[:4]!=b'Iout':raise ValueError('Bounded ImageJ ROI required')
    u=lambda i:struct.unpack_from('>H',raw,i)[0];integer=lambda i:struct.unpack_from('>i',raw,i)[0]
    if not 223<=u(4)<=229 or raw[6]!=1 or not u(50)&128:raise ValueError('Documented subpixel rectangle required')
    if u(16) or integer(36) or u(48) or u(54):raise ValueError('Unsupported ROI subtype')
    if integer(56) not in (0,1):raise ValueError('Unknown stack position')
    hdr=integer(60)
    if hdr:
        if not 64<=hdr<=len(raw)-36 or any(integer(hdr+i) not in (0,1) for i in (4,8,12)):raise ValueError('Unknown C/Z/T mapping')
    x,y,w,h=struct.unpack_from('>ffff',raw,18)
    if not all(math.isfinite(v) and 0<=v<=1_000_000 for v in (x,y,w,h)) or min(w,h)<=0:raise ValueError('Invalid subpixel bounds')
    # Official Roi(double,double,double,double,int) calls the integer constructor
    # with casts for x/y and Math.ceil for width/height; getBounds/contains use it.
    # In particular, do NOT round a stored20.0000038 width back to20.
    left,top=int(x),int(y);width,height=math.ceil(w),math.ceil(h)
    return {'version':u(4),'float_xywh':[x,y,w,h],'left':left,'top':top,'right':left+width,'bottom':top+height,'width':width,'height':height,'header_integer_bounds':list(struct.unpack_from('>hhhh',raw,8))}


def provided_target(size,positive,negative):
    w,h=size
    if not 0<w*h<=40_000_000:raise ValueError('Bounded native image required')
    target=np.full((h,w),255,dtype=np.uint8);same_overlap=0
    for label,boxes in ((1,positive),(0,negative)):
        if not 5<=len(boxes)<=16:raise ValueError('At least five provided patches per class required')
        for b in boxes:
            x,y,r,z=(b[k] for k in ('left','top','right','bottom'))
            if x<0 or y<0 or r>w or z>h or r<=x or z<=y:raise ValueError('Provided coordinates outside original TIFF')
            region=target[y:z,x:r]
            if np.any((region!=255)&(region!=label)):raise ValueError('Provided opposite-class patch conflict')
            same_overlap+=int((region==label).sum());region[:]=label
    return target,same_overlap
