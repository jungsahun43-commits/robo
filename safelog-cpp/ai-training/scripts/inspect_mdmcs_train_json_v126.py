"""One preregistered TRAIN annotation member; no image/mask decode or rasterization."""
from collections import Counter
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_hrcds_metadata_v121 import git, process_exists, sha, strict_json, write_json
from scripts.inspect_mdmcs_zip_directory_v124 import RejectRedirect, check_headers

SCRIPT = 'scripts/inspect_mdmcs_train_json_v126.py'
POLICY = ROOT / 'reports/industrial-mdmcs-v126-train-json-policy.json'
OUT = ROOT / 'data/industrial-mdmcs-v126-train-json'
RUN = ROOT / 'runs/industrial-mdmcs-v126-train-json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def local_header(raw, member, extra_limit):
    if len(raw) != 30:
        raise ValueError('Expected exactly 30 local-header bytes')
    sig, version, flags, method, mtime, mdate, crc, packed, expanded, nlen, xlen = struct.unpack('<4s5H3I2H', raw)
    if sig != b'PK\x03\x04' or version >= 45 or flags != member['flags'] or flags & (1 | 64 | 8192):
        raise ValueError('Local header signature/version/flags mismatch')
    if method != 8 or method != member['compression_method']:
        raise ValueError('Only the observed deflate member is allowed')
    expected_crc = int(member['CRC32_declared_not_verified'], 16)
    expected_packed, expected_expanded = member['compressed_bytes_declared'], member['expanded_bytes_declared']
    if flags & 8:
        if crc not in (0, expected_crc) or packed not in (0, expected_packed) or expanded not in (0, expected_expanded):
            raise ValueError('Descriptor-mode local declarations disagree with central directory')
    elif (crc, packed, expanded) != (expected_crc, expected_packed, expected_expanded):
        raise ValueError('Local sizes/CRC disagree with central directory')
    if nlen != len(base64.b64decode(member['filename_bytes_base64'], validate=True)) or xlen > extra_limit:
        raise ValueError('Unexpected local filename length or extra-field bound')
    return {'filename_bytes': nlen, 'extra_bytes': xlen, 'data_descriptor_flag': bool(flags & 8)}


def verify_local_name_extra(raw, header, member):
    nlen, xlen = header['filename_bytes'], header['extra_bytes']
    if len(raw) != nlen + xlen or raw[:nlen] != base64.b64decode(member['filename_bytes_base64'], validate=True):
        raise ValueError('Local filename bytes differ from frozen central directory')
    extra, cursor = raw[nlen:], 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise ValueError('Truncated local extra field')
        tag, length = struct.unpack_from('<HH', extra, cursor)
        cursor += 4
        if cursor + length > len(extra) or tag in (0x0001, 0x7075):
            raise ValueError('ZIP64/alternate Unicode path/malformed local extra rejected')
        cursor += length


def expand_annotation(packed, expected_bytes, expected_crc):
    inflater = zlib.decompressobj(-15)
    raw = inflater.decompress(packed, expected_bytes + 1)
    if len(raw) != expected_bytes or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
        raise ValueError('Bounded deflate length/stream check failed')
    crc = zlib.crc32(raw) & 0xffffffff
    if crc != int(expected_crc, 16):
        raise ValueError('Selected annotation member CRC mismatch; original preserved')
    return raw


def schema_observations(document, expected_image_basename, max_shapes):
    if not isinstance(document, dict):
        raise ValueError('Annotation JSON root must be an object')
    issues = []
    dimensions = [document.get('imageWidth'), document.get('imageHeight')]
    dimensions_valid = all(type(v) is int and 0 < v <= 32768 for v in dimensions)
    if not dimensions_valid:
        issues.append('Missing or unexpected declared image dimensions')
    image_path = document.get('imagePath')
    basename = image_path.replace('\\', '/').rsplit('/', 1)[-1] if isinstance(image_path, str) else None
    if basename != expected_image_basename:
        issues.append('Declared imagePath basename does not match reserved image filename')
    payload = document.get('imageData')
    encoded_image_present = isinstance(payload, str) and len(payload) > 0
    if payload is not None and not isinstance(payload, str):
        issues.append('Unexpected imageData type; no payload interpretation')
    shapes = document.get('shapes')
    if not isinstance(shapes, list):
        issues.append('Expected shapes list absent')
        shapes = []
    if len(shapes) > max_shapes:
        raise ValueError('Annotation shape-count bound exceeded')
    labels, shape_types, group_types = Counter(), Counter(), Counter()
    point_count = 0
    invalid_points = 0
    bounds_failures = 0
    coords = []
    for shape in shapes:
        if not isinstance(shape, dict):
            issues.append('Nonobject shape observed')
            continue
        label, shape_type = shape.get('label'), shape.get('shape_type')
        labels[label if isinstance(label, str) and len(label) <= 256 else '<unexpected-label-type-or-length>'] += 1
        shape_types[shape_type if isinstance(shape_type, str) and len(shape_type) <= 64 else '<missing-or-unexpected-shape-type>'] += 1
        group_types[type(shape.get('group_id')).__name__] += 1
        points = shape.get('points')
        if not isinstance(points, list):
            issues.append('Shape points list absent')
            continue
        for point in points:
            point_count += 1
            if not isinstance(point, list) or len(point) != 2 or not all(
                type(v) in (int, float) and math.isfinite(v) for v in point):
                invalid_points += 1
                continue
            coords.append(point)
            if dimensions_valid and not (0 <= point[0] <= dimensions[0] and 0 <= point[1] <= dimensions[1]):
                bounds_failures += 1
    flags = document.get('flags')
    return {'root_keys_and_types': {key: type(value).__name__ for key, value in document.items()},
        'declared_version_if_short_string': document.get('version') if isinstance(document.get('version'), str) and len(document['version']) <= 64 else None,
        'declared_image_width_height': dimensions, 'declared_dimensions_are_positive_ints': dimensions_valid,
        'imagePath_string_bytes': len(image_path.encode('utf-8')) if isinstance(image_path, str) else None,
        'imagePath_basename_matches_expected': basename == expected_image_basename,
        'imagePath_followed_or_used_for_filesystem_access': False,
        'imageData_present': 'imageData' in document, 'imageData_type': type(payload).__name__,
        'nonempty_encoded_image_payload_acquired_in_annotation': encoded_image_present,
        'encoded_imageData_character_count': len(payload) if isinstance(payload, str) else None,
        'encoded_imageData_UTF8_bytes': len(payload.encode('utf-8')) if isinstance(payload, str) else None,
        'encoded_imageData_string_SHA256_not_decoded_photo_SHA': hashlib.sha256(payload.encode('utf-8')).hexdigest() if encoded_image_present else None,
        'imageData_decoded_or_displayed': False,
        'root_flag_key_types_only': {str(k): type(v).__name__ for k, v in flags.items()} if isinstance(flags, dict) else None,
        'shape_count': len(shapes), 'literal_label_shape_counts': dict(labels), 'literal_shape_type_counts': dict(shape_types),
        'annotation_group_id_type_counts_not_physical_groups': dict(group_types),
        'point_count': point_count, 'invalid_nonfinite_or_nonpair_point_count': invalid_points,
        'points_outside_closed_declared_image_edges': bounds_failures,
        'closed_edge_numeric_check_is_not_authoritative_raster_contract': True,
        'coordinate_min_xy': [min(p[k] for p in coords) for k in (0, 1)] if coords else None,
        'coordinate_max_xy': [max(p[k] for p in coords) for k in (0, 1)] if coords else None,
        'all_observed_shapes_explicit_polygon': bool(shapes) and shape_types == {'polygon': len(shapes)},
        'explicit_background_unknown_related_root_key_names_only': [k for k in document if any(token in k.lower() for token in ('background', 'unknown', 'ignore'))],
        'schema_observation_issues': sorted(set(issues)),
        'background_unknown_RGB_mask_palette_raster_and_physical_group_semantics_verified': False,
        'GT_modified_or_unannotated_pixels_declared_negative': False, 'ready_for_pixels_or_training': False}


def new_fixtures(member):
    raw = struct.pack('<4s5H3I2H', b'PK\x03\x04', 20, member['flags'], 8, 0, 0, 0, 0, 0,
                      len(base64.b64decode(member['filename_bytes_base64'])), 0)
    assert local_header(raw, member, 4096)['data_descriptor_flag']
    try:
        verify_local_name_extra(b'x' * len(base64.b64decode(member['filename_bytes_base64'])),
                                local_header(raw, member, 4096), member)
    except ValueError:
        pass
    else:
        raise AssertionError('Local path mismatch accepted')
    sample = b'{"imageData":null,"shapes":[]}'
    encoder = zlib.compressobj(wbits=-15)
    packed = encoder.compress(sample) + encoder.flush()
    assert expand_annotation(packed, len(sample), f'{zlib.crc32(sample) & 0xffffffff:08x}') == sample
    try:
        expand_annotation(packed, len(sample), '00000000')
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect selected-member CRC accepted')
    observed = schema_observations(strict_json('{"imageWidth":2,"imageHeight":2,"imagePath":"fixture.jpg","imageData":"uninterpreted-fixture","shapes":[]}'), 'fixture.jpg', 100)
    assert observed['nonempty_encoded_image_payload_acquired_in_annotation'] and not observed['imageData_decoded_or_displayed']
    try:
        strict_json('{"shapes":[],"shapes":[]}')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate annotation JSON keys accepted')
    return {'local_header_and_filename_guard': True, 'bounded_deflate_and_actual_CRC_guard': True,
            'strict_JSON_and_encoded_payload_not_decoded': True, 'old_completed_fixtures_rerun': False}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, digest in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/MDMCS_V126_TRAIN_JSON_PROTOCOL_KO.md',
                 'scripts/inspect_hrcds_metadata_v121.py', 'scripts/inspect_mdmcs_zip_directory_v124.py'):
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name), name
    assert not git('status', '--porcelain').strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert not process_exists(23948)
    assert not list(ROOT.glob('operation.lock')) and not list((ROOT / 'runs').rglob('operation.lock'))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    rows = strict_json((ROOT / policy['frozen_directory_declarations']).read_text(encoding='utf-8'))
    candidates = sorted(r['name'] for r in rows if r['name'].startswith('HRCDS/train_annotations/') and r['name'].endswith('.json'))
    assert candidates[0] == policy['selected_member']['name']
    member = next(r for r in rows if r['name'] == policy['selected_member']['name'])
    assert member == policy['selected_member']
    storage = strict_json((ROOT / policy['frozen_storage_HEAD']).read_text(encoding='utf-8'))
    assert storage['status'] == 200 and storage['requested_URL'] == policy['url']
    assert storage['headers']['ETag'] == policy['strong_ETag']
    assert storage['headers']['x-amz-version-id'] == policy['storage_version_id']
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'one-TRAIN-annotation-schema-only'})
    received, requests = 0, []
    try:
        test = [r['name'] for r in rows if r['name'].split('/')[1] in policy['publisher_TEST_literal_folders'] and not r['is_directory_name']]
        held = [r['name'] for r in rows if r['name'].split('/')[1] in policy['publisher_VAL_literal_folders'] and not r['is_directory_name']]
        assert len(test) == 400 and len(held) == 300 and member['name'] not in test + held
        write_json(OUT / 'reserved-TEST-VAL-and-all-possible-copies-before-annotation-content.json', {
            'UTC': utc(), 'publisher_TEST_literal_paths_reserved': test, 'publisher_VAL_literal_paths_HELD': held,
            'all_members_and_possible_copies_UNASSIGNED_protection_retained': [r['name'] for r in rows],
            'selected_TRAIN_member_only_schema_probe_no_role_promotion': member['name'],
            'photo_copy_identity_physical_groups_final_FIT_VAL_CAL_and_graph_verified': False})
        write_json(RUN / 'annotation-range-schema-fixtures.json', new_fixtures(member))
        opener = urllib.request.build_opener(RejectRedirect())

        def request(name, start, length):
            nonlocal received
            if length <= 0 or received + length > policy['aggregate_response_body_limit']:
                raise ValueError('Preregistered aggregate Range body bound rejected before request')
            interval = (start, start + length - 1)
            req = urllib.request.Request(policy['url'], headers={'User-Agent': 'SafeLog-academic-annotation-schema-review/1.0',
                'Accept-Encoding': 'identity', 'Range': f'bytes={interval[0]}-{interval[1]}', 'If-Range': policy['strong_ETag']})
            try:
                response = opener.open(req, timeout=45)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                receipt = {'UTC': utc(), 'URL': response.url, 'status': response.code, 'range': interval,
                    'If_Range': policy['strong_ETag'], 'headers': {k: v for k, v in response.headers.items()
                    if k.lower() not in ('cookie', 'set-cookie')}, 'body_bytes_read_by_helper': 0}
                write_json(OUT / (name + '-headers.json'), receipt)
                requests.append(receipt)
                check_headers(response.code, response.url, response.headers, policy['url'], policy['archive_total_bytes'], policy['strong_ETag'], interval)
                if response.headers.get('x-amz-version-id') != policy['storage_version_id']:
                    raise ValueError('Storage version changed; body not read')
                partial = OUT / (name + '.partial')
                count = 0
                with partial.open('xb') as output:
                    while count < length:
                        chunk = response.read(min(65536, length - count))
                        if not chunk:
                            raise ValueError('Truncated exact member Range; partial preserved')
                        output.write(chunk)
                        count += len(chunk)
                        received += len(chunk)
                        receipt['body_bytes_read_by_helper'] = count
                final = OUT / (name + '.bin')
                partial.rename(final)
                write_json(OUT / (name + '-receipt.json'), dict(receipt, local_SHA256=sha(final)))
                return final.read_bytes()

        start = member['local_header_offset_declared']
        fixed = request('selected-local-header30', start, 30)
        header = local_header(fixed, member, policy['max_local_extra_bytes'])
        variable = request('selected-local-name-extra', start + 30, header['filename_bytes'] + header['extra_bytes'])
        verify_local_name_extra(variable, header, member)
        payload_start = start + 30 + len(variable)
        next_offset = min(r['local_header_offset_declared'] for r in rows if r['local_header_offset_declared'] > start)
        if payload_start + member['compressed_bytes_declared'] > min(next_offset, policy['central_directory_offset']):
            raise ValueError('Selected compressed payload would cross next member/directory')
        packed = request('selected-annotation-deflate', payload_start, member['compressed_bytes_declared'])
        raw = expand_annotation(packed, member['expanded_bytes_declared'], member['CRC32_declared_not_verified'])
        raw_path = OUT / 'publisher-train_0001.json'
        with raw_path.open('xb') as output:
            output.write(raw)
        observations = schema_observations(strict_json(raw.decode('utf-8')), policy['expected_image_basename'], policy['max_shapes'])
        write_json(OUT / 'annotation-schema-observations.json', observations)
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes'] - 1048576
        result = {'UTC': utc(), 'status': 'ONE_TRAIN_ANNOTATION_SCHEMA_COMPLETE', 'preregistered_HEAD': head,
            'policy_SHA256': sha(POLICY), 'selected_member': member['name'], 'requests': requests,
            'actual_new_range_body_bytes': received, 'source_annotation_JSON_members_acquired': 1,
            'selected_member_encoded_bytes': len(packed), 'selected_member_expanded_bytes': len(raw),
            'selected_member_CRC32_verified': member['CRC32_declared_not_verified'], 'selected_raw_JSON_SHA256': sha(raw_path),
            'schema_observations': observations, 'publisher_TEST_content_or_VAL_content_accessed': 0,
            'original_metadata_HEAD_CD_or_completed_jobs_fixtures_rerun': False,
            'full_archive_SHA_all_member_CRC_native_GT_RGB_mask_semantics_groups_graph_or_minima_verified': False,
            'source_RGB_or_mask_files_downloaded_or_decoded': 0, 'embedded_image_payload_acquired_if_present': observations['nonempty_encoded_image_payload_acquired_in_annotation'],
            'source_visual_QA_rasterization_features_scores_gradients_optimizer_CAL_TEST_API_CPP_ZIP_APK': 0,
            'ready_for_pixels_or_training': False, 'stored_bytes_before_final_report': stored, 'disk_free': shutil.disk_usage(ROOT).free}
        write_json(OUT / 'annotation-schema-complete.json', result)
        print(json.dumps({'status': result['status'], 'range_bytes': received, 'annotation_JSON_SHA256': sha(raw_path),
            'shapes': observations['shape_count'], 'encoded_imageData_acquired': observations['nonempty_encoded_image_payload_acquired_in_annotation'],
            'ready': False}), flush=True)
    except BaseException as error:
        write_json(RUN / 'failure.json', {'UTC': utc(), 'error': repr(error), 'preregistered_HEAD': head,
            'requests': requests, 'actual_range_body_bytes': received, 'originals_partials_headers_preserved': True,
            'image_mask_decode_gradient_optimizer_CAL_TEST': 0, 'training_failure': False,
            'whole_stage_restart_retry_or_full_download_allowed': False, 'ready_for_training': False})
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
