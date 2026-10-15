"""Audit already extracted original XML; preserve invalid author coordinates."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha

OUT = ROOT / 'data/industrial-structured-metal-v42-source'


def annotation_metadata(raw):
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('DTD/entities not permitted in source XML')
    tree = ET.fromstring(raw)
    width = int(tree.findtext('size/width'))
    height = int(tree.findtext('size/height'))
    if not 0 < width <= 16384 or not 0 < height <= 16384:
        raise ValueError('Invalid declared source geometry')
    boxes = []
    for item in tree.findall('object'):
        errors = []
        name = item.findtext('name')
        if not name:
            errors.append('missing_class')
        try:
            xyxy = [int(item.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        except (TypeError, ValueError):
            xyxy = None
            errors.append('missing_or_noninteger_coordinate')
        if xyxy is not None:
            if not (0 <= xyxy[0] < xyxy[2] <= width):
                errors.append('invalid_x_bounds')
            if not (0 <= xyxy[1] < xyxy[3] <= height):
                errors.append('invalid_y_bounds')
        boxes.append({'class': name, 'xyxy': xyxy, 'coordinate_valid': not errors, 'errors': errors})
    return {'XML_filename': tree.findtext('filename'), 'declared_size': [width, height], 'boxes': boxes,
            'author_photo_label': int(bool(boxes)), 'all_boxes_coordinate_valid': all(box['coordinate_valid'] for box in boxes)}


def main():
    assert not (OUT / 'annotation-content-metadata.json').exists()
    headers = read(OUT / 'annotation-headers-original.json')['rows']
    reserved = read(OUT / 'reserved-author-split-before-photos.json')
    by_id = {row['id']: row for row in reserved['rows']}
    rows = []
    for item in headers:
        path = OUT / 'publisher-annotation-files' / item['name']
        data = path.read_bytes()
        assert len(data) == item['unpacked'] and zlib.crc32(data) == item['unpacked_CRC32'], item['name']
        key = Path(item['name']).stem
        role = by_id[key]
        rows.append({'id': key, 'publisher_role': role['publisher_role'], 'phase': role['phase'],
            'capture_timestamp_proxy': role['capture_timestamp_proxy'], 'annotation': path.relative_to(ROOT).as_posix(),
            'annotation_SHA256': hashlib.sha256(data).hexdigest(),
            'unpacked_CRC32_verified': f'{item["unpacked_CRC32"]:08x}', **annotation_metadata(data)})
    assert len(rows) == len(by_id) == 9239
    phases = sorted({row['phase'] for row in rows})
    invalid = [row for row in rows if not row['all_boxes_coordinate_valid']]
    report = {'UTC': datetime.now(timezone.utc).isoformat(), 'annotation_files': len(rows),
        'all_original_XML_size_and_unpacked_CRC32_verified': True,
        'counts_author_photo_labels': {phase: dict(Counter(str(row['author_photo_label']) for row in rows if row['phase'] == phase)) for phase in phases},
        'invalid_coordinate_photos_by_phase': {phase: sum(not row['all_boxes_coordinate_valid'] for row in rows if row['phase'] == phase) for phase in phases},
        'invalid_coordinate_photos': len(invalid),
        'invalid_boxes': sum(not box['coordinate_valid'] for row in rows for box in row['boxes']),
        'box_classes': dict(Counter(box['class'] for row in rows for box in row['boxes'])),
        'declared_sizes': dict(Counter(str(row['declared_size']) for row in rows)),
        'valid_coordinate_defect_photos_by_phase': {phase: sum(row['author_photo_label'] == 1 and row['all_boxes_coordinate_valid'] for row in rows if row['phase'] == phase) for phase in phases},
        'invalid_coordinates_fixed_or_relabelled': False, 'invalid_TEST_removed': False,
        'author_photo_presence_not_verified_pixel_GT_or_whole_machine_health': True,
        'image_pixels_extracted_or_decoded': 0, 'new_features_scores_optimizer': 0,
        'single_illumination_to_XML_geometry_not_verified': True, 'physical_part_independence_unverified': True,
        'annotation_prefix_SHA256': sha(OUT / 'annotation-prefix-original-rar.bin'),
        'reserved_author_split_SHA256': sha(OUT / 'reserved-author-split-before-photos.json'),
        'whole_original_archive_MD5_verified': False,
        'native_decompressor': 'Windows bsdtar3.8.9; new annotation-only RAR container preserving original header+compressed payload, no author code run',
        'initial_strict_audit_failure': 'first invalid XML 20240204_144153_Gige[1]d: [167,516,222,222] in600x600; no model work had started'}
    new(OUT / 'annotation-content-metadata.json', {'rows': rows, 'audit': report})
    new(OUT / 'invalid-coordinate-rows.json', invalid)
    report['annotation_metadata_SHA256'] = sha(OUT / 'annotation-content-metadata.json')
    new(ROOT / 'reports/industrial-structured-metal-v42-annotation-audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
