"""Input-only screw pose normalization and aspect-preserving connector views."""
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .equipment_reference_photo import ReferencePhotoMain
from .equipment_expansion import CATEGORIES, features_for_equipment, anomaly_map, photo_score
from .metal_reference import patch_features, nearest_cosine
from .component_detail import fine_features, whole_template_distance

ARCH = 'safelog-equipment-native-component-v6'
METHODS = {'ad_screw': ('aligned_hilo', 'aligned_fine'), 'loco_splicing_connectors': ('native_patch', 'native_fusion')}
VIEW_SIZE = {'ad_screw': (768, 256), 'loco_splicing_connectors': (1024, 512)}


class InputQualityError(ValueError): pass


def normalized(rgb):
    value = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()/255
    return (value-value.new_tensor([.485, .456, .406])[:, None, None])/value.new_tensor([.229, .224, .225])[:, None, None]


def screw_view(rgb):
    """No GT, predicted anomaly or category labels influence the object pose."""
    thumb = rgb.copy(); thumb.thumbnail((256, 256), Image.Resampling.BICUBIC)
    gray = np.array(thumb.convert('L')); threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if len(stats) < 2: raise InputQualityError('No isolated screw foreground; manual review required')
    index = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); yy, xx = np.where(labels == index)
    fraction = len(xx)/gray.size
    if not .02 <= fraction <= .60: raise InputQualityError('Ambiguous screw foreground area; manual review required')
    xy = np.stack(((xx+.5)*rgb.width/thumb.width-.5, (yy+.5)*rgb.height/thumb.height-.5), 1)
    center = xy.mean(0); eigenvalues, eigenvectors = np.linalg.eigh(np.cov((xy-center).T))
    ratio = float(eigenvalues[-1]/max(eigenvalues[0], 1e-6))
    if ratio < 3: raise InputQualityError('Foreground is not elongated enough; manual review required')
    major = eigenvectors[:, -1]; minor = np.array([-major[1], major[0]])
    projection = (xy-center)@major; across = (xy-center)@minor
    low, high = np.quantile(projection, [.20, .80])
    def width(values): return float(np.quantile(values, .95)-np.quantile(values, .05))
    if width(across[projection <= low]) < width(across[projection >= high]): major = -major
    minor = np.array([-major[1], major[0]])
    projection = (xy-center)@major; across = (xy-center)@minor
    p0, p1 = projection.min(), projection.max(); q0, q1 = across.min(), across.max()
    output_width, output_height = VIEW_SIZE['ad_screw']; scale = min(output_width*.88/(p1-p0), output_height*.88/(q1-q0))
    crop_center = center+major*(p0+p1)/2+minor*(q0+q1)/2
    a = major/scale; b = minor/scale; shift = crop_center-a*output_width/2-b*output_height/2
    affine = [float(a[0]), float(b[0]), float(shift[0]), float(a[1]), float(b[1]), float(shift[1])]
    values = np.array(rgb); border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1])); fill = tuple(int(x) for x in np.median(border, axis=0))
    aligned = rgb.transform((output_width, output_height), Image.Transform.AFFINE, affine, Image.Resampling.BICUBIC, fillcolor=fill)
    return aligned, {'method': 'input_otsu_largest_component_PCA_head_left', 'source_size': list(rgb.size), 'view_size': [output_width, output_height],
        'foreground_fraction': fraction, 'axis_ratio': ratio, 'otsu_threshold': float(threshold), 'source_from_view_affine': affine,
        'source_center': crop_center.tolist(), 'major_axis': major.tolist(), 'minor_axis': minor.tolist(), 'scale': float(scale),
        'GT_or_anomaly_predictions_used': False}


def component_view(rgb, category):
    if category == 'ad_screw': return screw_view(rgb)
    if category == 'loco_splicing_connectors':
        # Exact dataset2:1 aspect ratio is retained. Other aspect ratios are
        # letterboxed without stretching; supplied input remains unlabelled.
        width, height = VIEW_SIZE[category]; factor = min(width/rgb.width, height/rgb.height)
        resized = rgb.resize((max(1, round(rgb.width*factor)), max(1, round(rgb.height*factor))), Image.Resampling.BICUBIC)
        view = Image.new('RGB', (width, height), (127, 127, 127)); offset = ((width-resized.width)//2, (height-resized.height)//2); view.paste(resized, offset)
        return view, {'method': 'native_aspect_letterbox', 'source_size': list(rgb.size), 'view_size': [width, height], 'scale': factor, 'offset': list(offset), 'GT_or_anomaly_predictions_used': False}
    raise ValueError('Explicit supported native component required')


def descriptor(features, method):
    return fine_features(features) if method == 'aligned_fine' else patch_features(features)


def layout_features(features):
    return F.normalize(F.adaptive_avg_pool2d(patch_features(features), (16, 32)), dim=1, eps=1e-6)


def native_evidence(features, bank, method, layout_bank=None):
    value = descriptor(features, method); b, channels, height, width = value.shape
    distance = nearest_cosine(value.permute(0, 2, 3, 1).reshape(-1, channels), bank, query_chunk=2048, reference_chunk=8192)
    native_map = distance.reshape(b, height, width)
    size = (256, 768) if method.startswith('aligned_') else (512, 1024)
    local_map = F.interpolate(native_map[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
    local_score = photo_score(local_map)
    if method != 'native_fusion': return local_map, local_score, None
    if layout_bank is None: raise ValueError('Coherent whole-photo templates required for fusion')
    layout_map, layout_score = whole_template_distance(layout_features(features), layout_bank)
    return local_map, local_score, (layout_map, layout_score)


class NativeComponentMain(ReferencePhotoMain):
    def __init__(self, reference_counts, native_specs):
        super().__init__(reference_counts); self.native_specs = dict(native_specs)
        for category, specification in native_specs.items():
            method, count = specification['method'], specification['patches']
            if category not in METHODS or method not in METHODS[category] or type(count) is not int or count < 1:
                raise ValueError('Explicit supported native method and positive patch count required')
            channels = 41 if method == 'aligned_fine' else 1000
            self.register_buffer('native_reference_'+category, torch.zeros(count, channels, dtype=torch.float16))
            if method == 'native_fusion':
                self.register_buffer('native_layout_'+category, torch.zeros(specification['layout_patches'], 1000, dtype=torch.float16))

    def forward_native(self, original256, native, category):
        if category not in self.native_specs: raise ValueError('Native component not selected')
        # SAME backbone, two input views. The original256 location branch stays exact.
        location = super().forward_equipment(original256, category)
        specification = self.native_specs[category]; features = self.backbone(native)
        heat, local_score, layout = native_evidence(features, getattr(self, 'native_reference_'+category), specification['method'], getattr(self, 'native_layout_'+category, None))
        if layout is None: return location, heat, local_score, {'local_score': local_score}
        _, layout_score = layout
        score = torch.maximum(local_score/specification['local_CAL_median'], layout_score/specification['layout_CAL_median'])
        return location, heat, score, {'local_score': local_score, 'layout_score': layout_score}
