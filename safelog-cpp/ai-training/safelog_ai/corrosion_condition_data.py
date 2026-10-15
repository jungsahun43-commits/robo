"""Publisher class verification and immutable group allocation, without GT repair."""
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw

LABELS = {'_background_': 0, '2_Fair_Steel_Corrosion': 1,
          '3_Poor_Steel_Corrosion': 2, '4_Severe_Steel_Corrosion': 3,
          '__ignore__': 255}
RGB = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2,
       (128, 128, 0): 3, (224, 224, 192): 255}
DEVELOPMENT = {'fit', 'validation', 'cal'}


def native_classes(image):
    """Accept only documented VOC indices/colors; no grayscale threshold or repair."""
    if image.mode in ('P', 'L', 'I', 'I;16'):
        value = np.asarray(image).astype(np.int64)
        if not set(np.unique(value)) <= set(LABELS.values()):
            raise ValueError('Undocumented mask indices')
        if image.mode == 'P':
            palette = image.getpalette()
            for index in np.unique(value):
                color = tuple(palette[3*index:3*index+3])
                if RGB.get(color) != index:
                    raise ValueError('Palette differs from documented VOC class colors')
        return value.astype(np.uint8)
    if image.mode not in ('RGB', 'RGBA'):
        raise ValueError('Unsupported native mask mode')
    if image.mode == 'RGBA' and np.any(np.asarray(image)[..., 3] != 255):
        raise ValueError('Transparent mask has no documented class contract')
    rgb = np.asarray(image.convert('RGB'))
    result = np.full(rgb.shape[:2], 254, dtype=np.uint8)
    for color, index in RGB.items():
        result[np.all(rgb == color, axis=-1)] = index
    if np.any(result == 254):
        raise ValueError('Undocumented RGB mask colors')
    return result


def polygon_classes(annotation, size):
    """Independent audit raster only; it never replaces a provided annotation."""
    w, h = size
    if (annotation['imageWidth'], annotation['imageHeight']) != (w, h):
        raise ValueError('JSON/image native coordinate sizes differ')
    result = np.zeros((h, w), dtype=np.uint8)
    for shape in annotation['shapes']:
        if shape['label'] not in LABELS or shape.get('shape_type', 'polygon') != 'polygon':
            raise ValueError('Undocumented label or nonpolygon shape')
        points = np.asarray(shape['points'], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Invalid polygon coordinates')
        if np.any(points < -1) or np.any(points[:, 0] > w) or np.any(points[:, 1] > h):
            raise ValueError('Polygon outside native boundary tolerance of one pixel')
        mask = Image.new('1', size)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in points], fill=1, outline=1)
        result[np.asarray(mask, dtype=bool)] = LABELS[shape['label']]
    return result


def allocate(keys, held, prior_roles):
    result = {}; unlocked = []
    for key in sorted(set(keys)):
        roles = set(prior_roles.get(key, ()))
        if not roles <= DEVELOPMENT:
            raise ValueError('Undocumented development role')
        if key in held:
            result[key] = 'excluded_TEST_connected'
        elif len(roles) > 1:
            result[key] = 'excluded_prior_role_conflict'
        elif roles:
            result[key] = next(iter(roles))
        else:
            unlocked.append(key)
    ordered = sorted(unlocked, key=lambda k: (hashlib.sha256(('ccs66:'+k).encode()).hexdigest(), k))
    n = math.ceil(.15*len(ordered))
    for i, key in enumerate(ordered):
        result[key] = 'validation' if i < n else 'cal' if i < 2*n else 'fit'
    return result
