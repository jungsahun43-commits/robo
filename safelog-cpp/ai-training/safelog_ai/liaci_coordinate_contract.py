"""Numeric author-mask pairing probe only; no model, whole-photo label or GT repair."""
from io import BytesIO
import hashlib,math,struct,zlib
import numpy as np
from PIL import Image,ImageDraw


def read_binary_member(row,getter,limit):
    if row['directory'] or max(row['packed'],row['bytes'])>limit:raise ValueError('Member budget')
    offset=row['local_header_offset'];head=getter(offset,30)
    f=struct.unpack('<4s5H3I2H',head)
    if f[0]!=b'PK\x03\x04' or f[2]!=row['flags'] or f[3]!=row['method']:raise ValueError('Header identity')
    crc,packed,size,n,extra=f[6:]
    if f[2]&8:
        if crc not in (0,int(row['CRC32_declared'],16)) or packed not in (0,row['packed']) or size not in (0,row['bytes']):raise ValueError('Descriptor identity')
    elif (f'{crc:08x}',packed,size)!=(row['CRC32_declared'],row['packed'],row['bytes']):raise ValueError('Header size/CRC')
    name=getter(offset+30,n+extra)[:n].decode('utf-8' if f[2]&0x800 else 'cp437')
    if name!=row['member']:raise ValueError('Header filename')
    encoded=getter(offset+30+n+extra,row['packed'])
    if row['method']==0:raw=encoded
    elif row['method']==8:
        d=zlib.decompressobj(-15);raw=d.decompress(encoded,row['bytes']+1)
        if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('Deflate bound')
    else:raise ValueError('Unsupported compression')
    if len(raw)!=row['bytes'] or f'{zlib.crc32(raw):08x}'!=row['CRC32_declared']:raise ValueError('Member size/CRC')
    return raw


def checked_polygons(annotations,width,height):
    polygons=[]
    for row in annotations:
        if row['iscrowd']!=0 or not isinstance(row['segmentation'],list):raise ValueError('Unsupported COCO shape')
        for poly in row['segmentation']:
            if not isinstance(poly,list) or len(poly)<6 or len(poly)%2:raise ValueError('Invalid polygon')
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in poly):raise ValueError('Nonfinite coordinates')
            points=list(zip(poly[::2],poly[1::2]))
            if not all(0<=x<=width and 0<=y<=height for x,y in points):raise ValueError('Out-of-canvas coordinates; no guessed clipping')
            polygons.append(points)
    return polygons


def numeric_pair(rgb_bytes,mask_bytes,size,annotations):
    width,height=size
    if width*height>40_000_000 or max(len(rgb_bytes),len(mask_bytes))>32*1024**2:raise ValueError('Decode bound')
    with Image.open(BytesIO(rgb_bytes)) as im:
        if im.format!='JPEG' or getattr(im,'n_frames',1)!=1 or im.mode!='RGB' or list(im.size)!=list(size) or im.getexif().get(274) not in (None,1):raise ValueError('RGB format/size/orientation')
        rgb=im.copy()
        exif=im.getexif()
        capture={'camera_model':exif.get(272),'DateTimeOriginal':exif.get(36867),'DateTime':exif.get(306)}
    with Image.open(BytesIO(mask_bytes)) as im:
        if im.format!='BMP' or getattr(im,'n_frames',1)!=1 or im.mode!='1' or list(im.size)!=list(size):raise ValueError('Native binary BMP/coordinate contract')
        mask=np.asarray(im.convert('L'))
    if not set(np.unique(mask)).issubset({0,255}):raise ValueError('Unexpected binary values')
    positive=mask==255
    polygons=checked_polygons(annotations,width,height)
    if bool(polygons)!=bool(positive.any()):raise ValueError('Provided mask vs COCO category presence differs')
    reference=Image.new('1',(width,height));draw=ImageDraw.Draw(reference)
    for poly in polygons:draw.polygon(poly,fill=1)
    ref=np.asarray(reference,dtype=bool)
    # This is a named independent diagnostic representation, never substituted
    # for the publisher bitmap or claimed to reproduce an unknown exporter.
    result={'size':list(size),'RGB_SHA256':hashlib.sha256(rgb.tobytes()).hexdigest(),
            'native_mask_SHA256':hashlib.sha256(positive.tobytes()).hexdigest(),
            'native_positive_pixels':int(positive.sum()),'COCO_polygon_count':len(polygons),
            'PIL_reference_disagreement_pixels':int(np.count_nonzero(positive!=ref)),
            'PIL_reference_is_not_proven_publisher_exporter':True,'capture_metadata':capture}
    return rgb,positive,result
