"""Explicit publisher-normal regions; never invent whole-photo belt health."""
import math
import re
import numpy as np
from PIL import Image

LABELS={'objects','belt splice','deep scratch','longitudinal tear','normal belt','slight scratch','worn surface'}


def parent(name):
    return re.split(r'\.rf\.[0-9a-f]+',name,flags=re.I)[0].lower().replace('_jpg','.jpg').replace('_png','.png')


def provisional_split(name,publisher_split):
    # These are frame-number hints, not verified video or independent site IDs.
    match=re.fullmatch(r'frame_(\d+)\.(jpg|png)',parent(name))
    if not match: return 'excluded_unknown_group'
    index=int(match.group(1))
    if index<=149: return 'fit' if publisher_split=='train' else 'excluded_publisher_holdout'
    if 180<=index<=239: return 'cal'
    if index>=270: return 'reserved'
    return 'excluded_temporal_gap'


def normal_core(box,others,width,height,min_side=64):
    x,y,w,h=map(float,box)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0:
        raise ValueError('Finite positive normal box required')
    side=math.floor(.6*min(w,h)); l=math.ceil(x+w/2-side/2); t=math.ceil(y+h/2-side/2)
    r,b=l+side,t+side
    if side<min_side or not (0<=l<r<=width and 0<=t<b<=height): return None
    for label,other in others:
        if label not in LABELS: raise ValueError('Unknown publisher category')
        if label=='normal belt': continue
        ox,oy,ow,oh=map(float,other)
        if max(l,ox)<min(r,ox+ow) and max(t,oy)<min(b,oy+oh): return None
    return (l,t,r,b)


def fingerprints(image):
    """Rotation/mirror-aware pHash hints; similarity never proves source identity."""
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    x=np.arange(32); k=np.arange(8)[:,None]
    basis=np.cos(np.pi*(2*x+1)*k/64)*math.sqrt(2/32); basis[0]/=math.sqrt(2)
    hashes=[]
    for flip in (False,True):
        values=np.fliplr(gray) if flip else gray
        for turns in range(4):
            dct=(basis@np.rot90(values,turns)@basis.T).flatten(); bits=dct>np.median(dct[1:]); bits[0]=False
            hashes.append(f'{sum(int(v)<<i for i,v in enumerate(bits)):016x}')
    return sorted(set(hashes))


def distance(a,b):
    return min((int(x,16)^int(y,16)).bit_count() for x in a for y in b)


def grouped(rows,max_distance=6):
    """Conservative connected components for original names, bytes and pHash."""
    links=list(range(len(rows)))
    def find(i):
        while links[i]!=i: links[i]=links[links[i]]; i=links[i]
        return i
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            if a['parent']==b['parent'] or a['rgb_sha256']==b['rgb_sha256'] or distance(a['phash'],b['phash'])<=max_distance:
                x,y=find(i),find(j); links[max(x,y)]=min(x,y)
    return [find(i) for i in range(len(rows))]
