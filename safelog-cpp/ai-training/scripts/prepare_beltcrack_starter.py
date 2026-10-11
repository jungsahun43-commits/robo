"""Decode publisher-embedded belt frames and rasterize their actual polygons."""
import base64
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/equipment-beltcrack/starter-available-v1'
SIZE = 640


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(annotation, image):
    width, height = image.size
    assert width == annotation['imageWidth'] and height == annotation['imageHeight']
    scale = SIZE / max(width, height)
    w, h = round(width * scale), round(height * scale)
    left, top = (SIZE - w) // 2, (SIZE - h) // 2
    frame = Image.new('RGB', (SIZE, SIZE), (123, 116, 103))
    frame.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    mask = Image.new('L', (SIZE, SIZE))
    valid = Image.new('L', (SIZE, SIZE))
    ImageDraw.Draw(valid).rectangle((left, top, left+w-1, top+h-1), fill=255)
    draw = ImageDraw.Draw(mask)
    for shape in annotation['shapes']:
        assert shape['label'] == 'crack', 'Unknown label: do not silently relabel'
        points = shape['points']
        assert all(len(p) == 2 and all(math.isfinite(float(x)) for x in p) for p in points)
        transformed = [(p[0] * w / width + left, p[1] * h / height + top) for p in points]
        if shape['shape_type'] == 'polygon':
            assert len(points) >= 3
            draw.polygon(transformed, fill=255)
        elif shape['shape_type'] == 'rectangle':
            assert len(points) == 2
            xs, ys = zip(*transformed)
            draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=255)
        else:
            raise ValueError('Unsupported publisher annotation shape ' + shape['shape_type'])
    masked = np.asarray(mask) & np.asarray(valid)
    return frame, Image.fromarray(masked), valid


def main():
    assert not (DATA / 'manifest.json').exists(), 'Preserve completed manifest'
    acquisition = json.loads((DATA / 'acquisition.json').read_text(encoding='utf-8'))
    assert acquisition['status'] == 'complete' and acquisition['selected_members_extractor_checksum_passed']
    prepared = DATA / 'prepared'
    prepared.mkdir()
    rows, pixel_ids, exclusions = [], {}, []
    for i, entry in enumerate(acquisition['members']):
        path = DATA / 'original-annotations' / entry['name']
        assert sha(path) == entry['sha256']
        annotation = json.loads(path.read_text(encoding='utf-8'))
        assert annotation['imageData'], 'Publisher-embedded source photo required'
        raw = base64.b64decode(annotation['imageData'], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGB')
            pixels = np.asarray(image)
            pixel_id = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            if pixel_id in pixel_ids:
                # Abort for audit rather than leak identical images between splits.
                raise ValueError('Duplicate decoded embedded photo: ' + entry['name'] + ' / ' + pixel_ids[pixel_id])
            pixel_ids[pixel_id] = entry['name']
            frame, mask, valid = rasterize(annotation, image)
        item = {'id': entry['name'], 'split': entry['split'], 'sequence': entry['sequence'],
            'annotation_sha256': entry['sha256'], 'embedded_image_sha256': hashlib.sha256(raw).hexdigest(),
            'decoded_rgb_sha256': pixel_id, 'width': annotation['imageWidth'], 'height': annotation['imageHeight'],
            'publisher_crack_shapes': len(annotation['shapes']), 'photo_positive': bool(annotation['shapes']),
            'positive_pixels': int((np.asarray(mask) > 0).sum()),
            'valid_pixels': int((np.asarray(valid) > 0).sum())}
        for kind, value in [('image', frame), ('mask', mask), ('valid', valid)]:
            dest = prepared / (str(i) + '-' + kind + '.png')
            value.save(dest)
            item[kind] = dest.relative_to(DATA).as_posix()
            item[kind + '_sha256'] = sha(dest)
        if item['photo_positive'] and not item['positive_pixels']:
            raise ValueError('Positive annotation vanished at model resolution; audit instead of training as negative')
        rows.append(item)
        if i % 100 == 0:
            print('Belt decoded and polygon-audited ' + str(i), flush=True)
    groups = {s: sorted({r['sequence'] for r in rows if r['split'] == s}) for s in ('train', 'val')}
    assert not set(groups['train']) & set(groups['val'])
    counts = Counter((r['split'], 'positive' if r['photo_positive'] else 'annotated_empty') for r in rows)
    manifest = {'status': 'frozen', 'dataset': 'Official V1 archive named BeltCrack14ks, observed 9645 source images',
        'publication_count_mismatch': True, 'input_size': SIZE, 'rows': rows,
        'counts': {'/'.join(k): v for k, v in counts.items()}, 'sequences': groups,
        'photo_source': 'Author-embedded imageData in original LabelMe JSON; separate JPEG equivalence not asserted',
        'label_source': 'Publisher crack polygons/rectangles, no concrete or AI-generated labels',
        'pixel_truth': 'Publisher vertices rasterized on 640 letterboxed grid; approximate model-resolution geometry',
        'padding_excluded_from_loss_and_metrics': True, 'source_test_photos_or_labels_used': False,
        'decoded_exact_duplicate_photos': 0, 'independent_factory_or_physical_belt_verified': False,
        'selection_sha256': sha(DATA / 'selection.json'), 'acquisition_sha256': sha(DATA / 'acquisition.json'),
        'preparation_script_sha256': sha(Path(__file__))}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'counts': manifest['counts'], 'sequences': {s: len(g) for s, g in groups.items()}}), flush=True)


if __name__ == '__main__':
    main()
