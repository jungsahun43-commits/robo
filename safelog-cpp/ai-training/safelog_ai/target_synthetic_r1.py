"""Fixed stronger procedural marks. These labels describe edits, not natural faults."""
import cv2
import numpy as np
from PIL import Image
from .industrial_foreground import foreground_view
from .target_synthetic import source_residual


def prepare_background(rgb):
    view,foreground,geometry=foreground_view(rgb)
    fg=cv2.erode(foreground.numpy().astype('uint8'),np.ones((3,3),dtype='uint8'))>0
    if not fg.any():raise ValueError('No eroded input foreground for procedural placement')
    return np.asarray(view).copy(),fg,geometry


def synthesize_prepared(background,source,seed):
    base,fg,geometry=background;residual,shape,source_geometry=source
    rng=np.random.default_rng(seed);extent=int(rng.integers(32,97));ratio=extent/max(shape.shape)
    h,w=max(5,round(shape.shape[0]*ratio)),max(5,round(shape.shape[1]*ratio))
    angle=int(rng.choice([0,45,90,135]))
    values=Image.fromarray(residual.astype('float32')).resize((w,h),Image.Resampling.BILINEAR).rotate(angle,Image.Resampling.BILINEAR,expand=True)
    binary=Image.fromarray(shape*255).resize((w,h),Image.Resampling.NEAREST).rotate(angle,Image.Resampling.NEAREST,expand=True)
    residual=np.array(values);shape=np.asarray(binary)>0;h,w=shape.shape
    if int(shape.sum())<32:raise ValueError('Fewer than32 transformed operation pixels')
    points=np.argwhere(fg);placement=None
    for attempt in range(256):
        cy,cx=points[int(rng.integers(len(points)))];y,x=int(cy-h//2),int(cx-w//2)
        if min(x,y)<0 or x+w>512 or y+h>512:continue
        overlap=float(fg[y:y+h,x:x+w][shape].mean())
        if overlap>=.98:placement=x,y,overlap,attempt+1;break
    if placement is None:raise ValueError('No98percent eroded-colour-region placement in256 attempts')
    x,y,overlap,attempts=placement;amplitude=float(rng.uniform(24,64));sign=float(rng.choice([-1,1]));patch=base[y:y+h,x:x+w].astype('float32')
    output=None;actual=None
    for polarity in (sign,-sign):
        candidate=np.clip(patch+(residual*shape*amplitude*polarity)[...,None],0,255).astype('uint8')
        changed=np.any(candidate!=patch.astype('uint8'),axis=2)
        magnitude=np.abs(candidate.astype('int16')-patch.astype('int16')).mean(2)
        actual=(magnitude>=3)&shape
        if actual.sum()>=32 and float(magnitude[actual].mean())>=12:
            output=candidate;sign=polarity;break
    if output is None:raise ValueError('Fewer than32 visible pixels or mean RGB contrast below12')
    output[~actual]=base[y:y+h,x:x+w][~actual]
    result=base.copy();result[y:y+h,x:x+w]=output
    mask=np.zeros((512,512),dtype='uint8');mask[y:y+h,x:x+w]=actual
    changed=np.any(result!=base,axis=2);assert np.array_equal(changed,mask>0)
    return Image.fromarray(result),Image.fromarray(mask*255),{
        'seed':int(seed),'source_crop':source_geometry,'target_geometry':geometry,'target_rect_xywh':[x,y,w,h],
        'extent':extent,'short_axis_minimum':5,'angle':angle,'amplitude':amplitude,'polarity':sign,
        'placement_attempts':attempts,'input_eroded_colour_region_overlap':overlap,'colour_region_is_part_GT':False,
        'changed_pixels':int(actual.sum()),'mean_absolute_RGB_change':float(np.abs(result.astype('int16')-base.astype('int16'))[changed].mean()),
        'label_scope':'known procedural RGB modification support; not verified natural industrial fault GT',
        'new_natural_fault_labels':0}


def synthesize(rgb,source_rgb,source_mask,seed):
    return synthesize_prepared(prepare_background(rgb),source_residual(source_rgb,source_mask),seed)
