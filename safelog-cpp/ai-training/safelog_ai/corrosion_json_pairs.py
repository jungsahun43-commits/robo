"""Bounded author-JSON image/polygon pairs; never repairs a publisher PNG."""
import base64
import binascii
import io
import numpy as np
from PIL import Image, ImageDraw
from safelog_ai.corrosion_condition_data import LABELS

MAX_BYTES = 32*1024**2
MAX_PIXELS = 40_000_000
DEVELOPMENT = {'fit','validation','cal'}


def embedded_image(annotation):
    value = annotation.get('imageData')
    if not isinstance(value,str) or not value or len(value) > 4*((MAX_BYTES+2)//3)+1024**2:
        raise ValueError('Bounded embedded imageData required; no imagePath fallback')
    try:
        compact = value.translate(str.maketrans('', '', ' \t\r\n\v\f')).encode('ascii')
        if len(compact) > 4*((MAX_BYTES+2)//3): raise ValueError('Encoded image limit')
        raw = base64.b64decode(compact,validate=True)
    except (UnicodeError,binascii.Error) as error:
        raise ValueError('Invalid strict base64') from error
    if not raw or len(raw)>MAX_BYTES: raise ValueError('Decoded image limit')
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('JPEG','PNG') or getattr(im,'n_frames',1)!=1:
            raise ValueError('Only single-frame JPEG/PNG embedded images')
        if not 0<im.width*im.height<=MAX_PIXELS or im.getexif().get(274) not in (None,1):
            raise ValueError('Embedded image size or orientation unsupported')
        if (annotation.get('imageWidth'),annotation.get('imageHeight'))!=im.size:
            raise ValueError('JSON and embedded image native sizes differ')
        if im.mode not in ('RGB','RGBA','L'): raise ValueError('Undocumented image color mode')
        if im.mode=='RGBA' and np.any(np.asarray(im)[...,3]!=255):
            raise ValueError('Transparent embedded source image')
        fmt,mode,orientation=im.format,im.mode,im.getexif().get(274)
        rgb=im.convert('RGB')
        exif=im.getexif();sub=exif.get_ifd(34665) if exif.get(34665) else {}
        meta={'format':fmt,'original_mode':mode,'EXIF_orientation':orientation,
              'camera_model':exif.get(272),'capture_time':sub.get(36867),'encoded_bytes':len(raw)}
    return rgb,raw,meta


def polygon_labels(annotation,size):
    """Independent implementation of official4.5.2 ordered polygon class canvas.

    PIL clips finite out-of-canvas vertices as in the documented producer.
    Original coordinates are not moved/clamped. Ignore -1 is stored as255.
    """
    w,h=size
    if (annotation.get('imageWidth'),annotation.get('imageHeight'))!=(w,h):
        raise ValueError('JSON and embedded image size mismatch')
    shapes=annotation.get('shapes')
    if not isinstance(shapes,list) or len(shapes)>10000: raise ValueError('Bounded shapes list required')
    result=np.zeros((h,w),dtype=np.uint8);outside=0;total=0
    for shape in shapes:
        if shape.get('label') not in LABELS or shape.get('shape_type','polygon') not in (None,'polygon'):
            raise ValueError('Only documented polygon classes accepted')
        points=np.asarray(shape['points'],dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=2 or not 3<=len(points)<=100000 or not np.isfinite(points).all():
            raise ValueError('Finite polygon vertices required')
        total+=len(points)
        if total>1_000_000 or np.max(np.abs(points))>1_000_000: raise ValueError('Coordinate resource limit')
        outside+=int(np.count_nonzero((points[:,0]<0)|(points[:,0]>w-1)|(points[:,1]<0)|(points[:,1]>h-1)))
        mask=Image.fromarray(np.zeros((h,w),dtype=np.uint8))
        ImageDraw.Draw(mask).polygon([tuple(v) for v in points],outline=1,fill=1)
        result[np.asarray(mask,dtype=bool)]=LABELS[shape['label']]
    return result,{'vertices':total,'outside_canvas_vertices_clipped_by_rasterizer_not_moved':outside}


def locked_phase(publisher_role,held,roles):
    roles=set(roles)
    if not roles<=DEVELOPMENT: raise ValueError('Undocumented prior role')
    if publisher_role=='Test': return 'test'
    if publisher_role!='Train': raise ValueError('Unknown publisher role')
    if held:return 'excluded_TEST_connected'
    if len(roles)>1:return 'excluded_prior_role_conflict'
    if len(roles)!=1:raise ValueError('v66 role must be locked; no new allocation')
    return next(iter(roles))
