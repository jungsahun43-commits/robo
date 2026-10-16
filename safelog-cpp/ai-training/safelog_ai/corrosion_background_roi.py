"""Strict sparse ImageJ integer rectangle supervision; other pixels are unknown."""
import struct
import numpy as np
from PIL import Image


def rectangle(raw):
    if not 64<=len(raw)<=1024**2 or raw[:4]!=b'Iout':raise ValueError('Bounded ImageJ ROI required')
    short=lambda i:struct.unpack_from('>h',raw,i)[0]
    ushort=lambda i:struct.unpack_from('>H',raw,i)[0]
    integer=lambda i:struct.unpack_from('>i',raw,i)[0]
    version=ushort(4)
    if not 217<=version<=229 or raw[6]!=1:raise ValueError('Only documented integer rectangle ROI supported')
    if ushort(16)!=0 or integer(36)!=0 or ushort(48)!=0 or ushort(50)&128 or ushort(54)!=0:
        raise ValueError('Coordinates/composite/subtype/subpixel/rounded ROI unsupported')
    if integer(56) not in (0,1):raise ValueError('Unknown ROI stack position')
    hdr=integer(60)
    if hdr:
        if not 64<=hdr<=len(raw)-36:raise ValueError('Invalid second header')
        if any(integer(hdr+i) not in (0,1) for i in (4,8,12)):raise ValueError('Unknown ROI C/Z/T frame mapping')
    top,left,bottom,right=(short(i) for i in (8,10,12,14))
    if min(top,left)<0 or right<=left or bottom<=top:raise ValueError('Invalid native rectangle')
    # RoiDecoder constructs Roi(left,top,right-left,bottom-top), with half-open
    # integer rectangle membership. No one-based shift or inclusive last pixel.
    return {'version':version,'left':left,'top':top,'right':right,'bottom':bottom,'width':right-left,'height':bottom-top,'header2_offset':hdr}


def sparse_target(size,positive,negative):
    w,h=size
    if not 0<w*h<=40_000_000:raise ValueError('Bounded native size required')
    result=np.full((h,w),255,dtype=np.uint8)
    for label,boxes in ((1,positive),(0,negative)):
        if len(boxes)!=5:raise ValueError('Publisher five patches per class required')
        for box in boxes:
            x,y,r,b=(box[k] for k in ('left','top','right','bottom'))
            if r-x!=5 or b-y!=5 or x<0 or y<0 or r>w or b>h:raise ValueError('Native five by five patch required')
            region=result[y:b,x:r]
            if np.any(region!=255):raise ValueError('Overlapping provided patches require review')
            region[:]=label
    return result


def target512(target):
    h,w=target.shape;s=min(512/w,512/h);rw,rh=max(1,round(w*s)),max(1,round(h*s))
    left,top=(512-rw)//2,(512-rh)//2
    result=np.full((512,512),255,dtype=np.uint8)
    result[top:top+rh,left:left+rw]=np.asarray(Image.fromarray(target).resize((rw,rh),Image.Resampling.NEAREST))
    return result,{'native_size':[w,h],'resized_size':[rw,rh],'left':left,'top':top}
