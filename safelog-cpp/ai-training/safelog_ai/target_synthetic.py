"""Procedural marks for research augmentation, never verified natural defect GT."""
import cv2
import numpy as np
from PIL import Image
from .industrial_foreground import foreground_view

VERSION='safelog-target-synthetic-preview-v19'


def source_residual(rgb,truth):
    if rgb.size!=truth.size:raise ValueError('Original source/mask alignment required')
    gray=np.asarray(rgb.convert('L'));mask=(np.asarray(truth.convert('L'))>0).astype('uint8')
    if not mask.any():raise ValueError('Actual source shape must contain a FIT defect annotation')
    ys,xs=np.nonzero(mask);x0,x1=max(0,int(xs.min())-2),min(mask.shape[1],int(xs.max())+3);y0,y1=max(0,int(ys.min())-2),min(mask.shape[0],int(ys.max())+3)
    # This estimated clean background is not a captured clean ground-truth image.
    filled=cv2.inpaint(gray,mask*255,3,cv2.INPAINT_TELEA)
    residual=(gray.astype('float32')-filled.astype('float32'))*mask
    scale=float(np.quantile(np.abs(residual[mask>0]),.9))
    if scale<1:raise ValueError('Source has insufficient contrast for this procedure')
    return np.clip(residual[y0:y1,x0:x1]/scale,-1,1),mask[y0:y1,x0:x1],{'source_crop_xyxy':[x0,y0,x1,y1],'inpaint_estimate_not_clean_GT':True}


def synthesize(normal_rgb,source_rgb,source_mask,seed):
    rng=np.random.default_rng(seed);view,foreground,geometry=foreground_view(normal_rgb)
    base=np.asarray(view).copy();fg=foreground.numpy();residual,shape,source_geometry=source_residual(source_rgb,source_mask)
    extent=int(rng.integers(24,73));ratio=extent/max(shape.shape);h,w=max(2,round(shape.shape[0]*ratio)),max(2,round(shape.shape[1]*ratio))
    values=Image.fromarray(residual.astype('float32')).resize((w,h),Image.Resampling.BILINEAR)
    binary=Image.fromarray(shape*255).resize((w,h),Image.Resampling.NEAREST)
    angle=int(rng.choice([0,45,90,135]));values=values.rotate(angle,Image.Resampling.BILINEAR,expand=True);binary=binary.rotate(angle,Image.Resampling.NEAREST,expand=True)
    residual=np.array(values);shape=np.asarray(binary)>0;h,w=shape.shape
    if int(shape.sum())<16:raise ValueError('Shape too small after declared transformation')
    points=np.argwhere(fg);chosen=None
    for attempt in range(256):
        cy,cx=points[int(rng.integers(len(points)))];y=int(cy-h//2);x=int(cx-w//2)
        if x<0 or y<0 or x+w>512 or y+h>512:continue
        coverage=float(fg[y:y+h,x:x+w][shape].mean())
        if coverage>=.98:chosen=(x,y,coverage,attempt+1);break
    if chosen is None:raise ValueError('No placement inside input-estimated foreground after256 attempts')
    x,y,coverage,attempts=chosen;amplitude=float(rng.uniform(16,48));sign=float(rng.choice([-1,1]));patch=base[y:y+h,x:x+w].astype('float32')
    output=None;actual=None
    for polarity in (sign,-sign):
        delta=residual*shape*amplitude*polarity
        candidate=np.clip(patch+delta[...,None],0,255).astype('uint8')
        changed=np.max(np.abs(candidate.astype('int16')-patch.astype('int16')),axis=2)
        actual=(changed>=2)&shape
        if actual.sum()>=16 and float(changed[actual].mean())>=8:
            output=candidate;sign=polarity;break
    if output is None:raise ValueError('Procedural mark not sufficiently visible after clipping')
    # Only the actual operation support is labelled; this is still synthetic GT.
    result=base.copy();output[~actual]=base[y:y+h,x:x+w][~actual];result[y:y+h,x:x+w]=output
    mask=np.zeros((512,512),dtype=np.uint8);mask[y:y+h,x:x+w]=actual
    changed=np.any(result!=base,axis=2);assert np.array_equal(changed,mask>0)
    return Image.fromarray(base),Image.fromarray(result),Image.fromarray(mask*255),{
        'seed':int(seed),'source_geometry':source_geometry,'target_geometry':geometry,'target_rect_xywh':[x,y,w,h],
        'source_shape_extent':extent,'angle_degrees':angle,'amplitude':amplitude,'polarity':sign,'placement_attempts':attempts,
        'input_foreground_overlap':coverage,'input_foreground_is_ground_truth':False,
        'actual_operation_mask_pixels':int(actual.sum()),'mean_absolute_changed_intensity':float(np.abs(result.astype('int16')-base.astype('int16'))[changed].mean()),
        'mask_label_scope':'known procedural RGB modification support, not natural industrial defect annotation',
        'new_real_industrial_faults_collected':0,'human_verified_natural_fault':False}
