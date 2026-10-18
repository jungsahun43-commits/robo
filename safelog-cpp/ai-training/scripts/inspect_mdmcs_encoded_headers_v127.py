"""Two fixed original TRAIN encoded files; parse headers without image pixel decode."""
from datetime import datetime, timezone
import hashlib
import json
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
from scripts.inspect_mdmcs_train_json_v126 import local_header, verify_local_name_extra, expand_annotation as expand_zip_member

SCRIPT = 'scripts/inspect_mdmcs_encoded_headers_v127.py'
POLICY = ROOT / 'reports/industrial-mdmcs-v127-encoded-headers-policy.json'
OUT = ROOT / 'data/industrial-mdmcs-v127-encoded-headers'
RUN = ROOT / 'runs/industrial-mdmcs-v127-encoded-headers'


def utc():
    return datetime.now(timezone.utc).isoformat()


def jpeg_headers(raw):
    if not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9'):
        raise ValueError('Expected JPEG SOI and final EOI boundaries')
    cursor, segments, frame = 2, [], None
    sof_markers = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while cursor < len(raw) and len(segments) < 2048:
        if raw[cursor] != 255:
            raise ValueError('Nonmarker bytes before JPEG scan')
        while cursor < len(raw) and raw[cursor] == 255:
            cursor += 1
        if cursor + 3 > len(raw):
            raise ValueError('Truncated JPEG header')
        marker = raw[cursor]
        cursor += 1
        if marker in {0, 1, 0xd8, 0xd9, *range(0xd0, 0xd8)}:
            raise ValueError('Unexpected standalone JPEG marker before scan')
        length = struct.unpack_from('>H', raw, cursor)[0]
        if length < 2 or cursor + length > len(raw):
            raise ValueError('JPEG segment length exceeds encoded member')
        data = raw[cursor + 2:cursor + length]
        segments.append({'marker': f'ff{marker:02x}', 'segment_bytes_including_length': length})
        cursor += length
        if marker in sof_markers:
            if frame is not None or len(data) < 6:
                raise ValueError('Multiple/truncated JPEG frames are not supported')
            precision = data[0]
            height, width, count = struct.unpack_from('>HHB', data, 1)
            if len(data) != 6 + 3 * count or not (0 < width <= 32768 and 0 < height <= 32768 and 1 <= count <= 4):
                raise ValueError('JPEG SOF dimensions/component length outside fixed bounds')
            if not 2 <= precision <= 16:
                raise ValueError('Unexpected JPEG sample precision')
            components = [{'id': data[6 + 3*i], 'H': data[7 + 3*i] >> 4,
                           'V': data[7 + 3*i] & 15, 'table': data[8 + 3*i]} for i in range(count)]
            if len({c['id'] for c in components}) != count or any(not 1 <= c['H'] <= 4 or not 1 <= c['V'] <= 4 for c in components):
                raise ValueError('Unexpected JPEG SOF component declarations')
            frame = {'SOF_marker': f'ff{marker:02x}', 'width': width, 'height': height,
                     'bit_depth': precision, 'component_count': count, 'components': components}
        if marker == 0xda:
            if frame is None or not data:
                raise ValueError('JPEG scan before supported frame header')
            count = data[0]
            if not 1 <= count <= frame['component_count'] or len(data) != 4 + 2 * count:
                raise ValueError('Unexpected JPEG SOS header length')
            scan_ids = [data[1 + 2*i] for i in range(count)]
            if len(set(scan_ids)) != count or not set(scan_ids) <= {c['id'] for c in frame['components']}:
                raise ValueError('JPEG scan references undeclared components')
            return dict(frame, format='JPEG', pre_scan_segments=segments,
                bytes_after_first_SOS_header_not_interpreted=len(raw)-cursor,
                entropy_scan_decoded=False, EXIF_or_app_payload_executed_or_used_for_groups=False,
                RGB_color_meaning_or_GT_verified=False)
    raise ValueError('No supported JPEG SOF/SOS within header bound')


def png_headers(raw):
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('PNG signature mismatch')
    cursor, chunks, ihdr, palette, transparency = 8, [], None, None, None
    idat_count, idat_bytes, idat_closed = 0, 0, False
    while cursor < len(raw) and len(chunks) < 2048:
        if cursor + 12 > len(raw):
            raise ValueError('Truncated PNG chunk header')
        length = struct.unpack_from('>I', raw, cursor)[0]
        kind = raw[cursor+4:cursor+8]
        if length > len(raw) - cursor - 12 or not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind) or kind[2] & 32:
            raise ValueError('PNG chunk length/type exceeds fixed encoded file')
        data = raw[cursor+8:cursor+8+length]
        expected = struct.unpack_from('>I', raw, cursor+8+length)[0]
        if zlib.crc32(kind + data) & 0xffffffff != expected:
            raise ValueError('PNG encoded chunk CRC mismatch')
        name = kind.decode('ascii')
        chunks.append({'type': name, 'bytes': length, 'encoded_chunk_CRC_verified': True})
        cursor += 12 + length
        if ihdr is None and kind != b'IHDR':
            raise ValueError('IHDR must be the first PNG chunk')
        if kind in (b'acTL', b'fcTL', b'fdAT'):
            raise ValueError('Animated PNG is outside the fixed static header probe')
        if kind == b'IHDR':
            if ihdr is not None or len(data) != 13:
                raise ValueError('Duplicate/malformed PNG IHDR')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', data)
            depths = {0:{1,2,4,8,16}, 2:{8,16}, 3:{1,2,4,8}, 4:{8,16}, 6:{8,16}}
            if not (0 < width <= 32768 and 0 < height <= 32768) or depth not in depths.get(color, set()) or compression != 0 or filtering != 0 or interlace not in (0,1):
                raise ValueError('Unsupported PNG IHDR declaration')
            ihdr = {'width': width, 'height': height, 'bit_depth': depth, 'color_type': color,
                    'compression': compression, 'filter_method': filtering, 'interlace': interlace}
        elif kind == b'PLTE':
            if palette is not None or idat_count or not 3 <= len(data) <= 768 or len(data) % 3 or ihdr['color_type'] in (0,4):
                raise ValueError('Unexpected PNG palette position/length/type')
            palette = [list(data[i:i+3]) for i in range(0, len(data), 3)]
            if ihdr['color_type'] == 3 and len(palette) > 2 ** ihdr['bit_depth']:
                raise ValueError('PNG palette exceeds declared index bit depth')
        elif kind == b'tRNS':
            if transparency is not None or idat_count:
                raise ValueError('Unexpected duplicate/post-IDAT transparency')
            color = ihdr['color_type']
            if (color == 3 and (palette is None or not 0 < len(data) <= len(palette))) or (color in (0,2) and len(data) != (2 if color == 0 else 6)) or color in (4,6):
                raise ValueError('PNG transparency length/color type mismatch')
            transparency = {'bytes': len(data), 'hex_declared_samples_or_alpha_only': data.hex()}
        elif kind == b'IDAT':
            if idat_closed or (ihdr['color_type'] == 3 and palette is None):
                raise ValueError('Noncontiguous PNG IDAT or missing palette')
            idat_count += 1
            idat_bytes += length
        elif kind == b'IEND':
            if len(data) or cursor != len(raw) or idat_count == 0:
                raise ValueError('PNG IEND/IDAT/trailing bytes mismatch')
            return dict(ihdr, format='PNG', chunks=chunks, PLTE_RGB_entries_not_class_mapping=palette,
                tRNS_declarations=transparency, IDAT_chunks=idat_count, IDAT_encoded_bytes=idat_bytes,
                IDAT_decompressed=False, pixel_values_histogram_raster_or_palette_class_meaning_verified=False)
        elif not kind[0] & 32:
            raise ValueError('Unknown critical PNG chunk')
        if idat_count and kind != b'IDAT':
            idat_closed = True
    raise ValueError('PNG IEND missing or chunk count bound exceeded')


def inspect_encoded(raw):
    if raw.startswith(b'\xff\xd8'):
        return jpeg_headers(raw)
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        return png_headers(raw)
    raise ValueError('Only JPEG/PNG magic is allowed; no decoder fallback')


def new_fixtures():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data) & 0xffffffff)
    fixture_png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 3, 0, 0, 0)) + chunk(b'PLTE', b'\x00\x00\x00\xff\x00\x00') + chunk(b'IDAT', b'uninterpreted-not-inflated-fixture') + chunk(b'IEND', b'')
    assert png_headers(fixture_png)['IDAT_decompressed'] is False
    damaged = bytearray(fixture_png)
    damaged[29] ^= 1
    try:
        png_headers(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('Bad PNG chunk CRC accepted')
    sof = b'\x08' + struct.pack('>HHB', 2, 2, 3) + b'\x01\x11\x00\x02\x11\x00\x03\x11\x00'
    sos = b'\x03\x01\x00\x02\x00\x03\x00\x00\x3f\x00'
    fixture_jpeg = b'\xff\xd8\xff\xc0' + struct.pack('>H', len(sof)+2) + sof + b'\xff\xda' + struct.pack('>H', len(sos)+2) + sos + b'uninterpreted-fixture\xff\xd9'
    assert jpeg_headers(fixture_jpeg)['entropy_scan_decoded'] is False
    for bad in (fixture_jpeg[:10], b'not an image header'):
        try:
            inspect_encoded(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated/unexpected header accepted')
    return {'PNG_header_palette_CRC_without_IDAT_inflate': True, 'JPEG_SOF_SOS_without_scan_decode': True,
            'truncated_or_unknown_encoding_rejected': True, 'old_completed_fixtures_rerun': False}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, digest in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/MDMCS_V127_ENCODED_HEADERS_PROTOCOL_KO.md',
                 'scripts/inspect_hrcds_metadata_v121.py', 'scripts/inspect_mdmcs_zip_directory_v124.py', 'scripts/inspect_mdmcs_train_json_v126.py'):
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name), name
    assert not git('status', '--porcelain').strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert not process_exists(23948)
    assert not list(ROOT.glob('operation.lock')) and not list((ROOT / 'runs').rglob('operation.lock'))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    rows = strict_json((ROOT / policy['frozen_directory']).read_text(encoding='utf-8'))
    for selected in policy['members']:
        assert next(r for r in rows if r['name'] == selected['central']['name']) == selected['central']
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'two-fixed-TRAIN-encoded-headers-only'})
    received, requests, completed = 0, [], []
    try:
        original = strict_json((ROOT / policy['frozen_reservation']).read_text(encoding='utf-8'))
        assert len(original['publisher_TEST_literal_paths_reserved']) == 400 and len(original['publisher_VAL_literal_paths_HELD']) == 300
        write_json(OUT / 'protection-before-encoded-content.json', dict(original, UTC=utc(),
            selected_member_names=[s['central']['name'] for s in policy['members']],
            no_FIT_VAL_CAL_photo_identity_or_physical_groups_promoted=True))
        write_json(RUN / 'encoded-header-fixtures.json', new_fixtures())
        opener = urllib.request.build_opener(RejectRedirect())

        def request(name, start, length):
            nonlocal received
            if length <= 0 or received + length > policy['aggregate_response_body_limit']:
                raise ValueError('Encoded-member aggregate Range bound exceeded before request')
            interval = (start, start + length - 1)
            req = urllib.request.Request(policy['url'], headers={'User-Agent': 'SafeLog-academic-encoded-header-review/1.0',
                'Accept-Encoding': 'identity', 'Range': f'bytes={interval[0]}-{interval[1]}', 'If-Range': policy['strong_ETag']})
            try:
                response = opener.open(req, timeout=45)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                receipt = {'UTC': utc(), 'URL': response.url, 'status': response.code, 'range': interval,
                    'If_Range': policy['strong_ETag'], 'headers': {k:v for k,v in response.headers.items() if k.lower() not in ('cookie','set-cookie')},
                    'body_bytes_read_by_helper': 0}
                write_json(OUT / (name+'-headers.json'), receipt)
                requests.append(receipt)
                check_headers(response.code, response.url, response.headers, policy['url'], policy['archive_total_bytes'], policy['strong_ETag'], interval)
                if response.headers.get('x-amz-version-id') != policy['storage_version_id']:
                    raise ValueError('Storage version changed; body not read')
                partial = OUT / (name+'.partial')
                count = 0
                with partial.open('xb') as output:
                    while count < length:
                        block = response.read(min(65536,length-count))
                        if not block:
                            raise ValueError('Truncated selected Range; partial preserved')
                        output.write(block)
                        count += len(block)
                        received += len(block)
                        receipt['body_bytes_read_by_helper'] = count
                final = OUT / (name+'.bin')
                partial.rename(final)
                write_json(OUT / (name+'-receipt.json'), dict(receipt, local_SHA256=sha(final)))
                return final.read_bytes()

        for selected in policy['members']:
            member, tag = selected['central'], selected['tag']
            start = member['local_header_offset_declared']
            fixed = request(tag+'-local-header30',start,30)
            header = local_header(fixed,member,policy['max_local_extra_bytes'])
            variable = request(tag+'-local-name-extra',start+30,header['filename_bytes']+header['extra_bytes'])
            verify_local_name_extra(variable,header,member)
            payload_start = start+30+len(variable)
            next_offset = min(r['local_header_offset_declared'] for r in rows if r['local_header_offset_declared'] > start)
            if payload_start+member['compressed_bytes_declared'] > min(next_offset,policy['central_directory_offset']):
                raise ValueError('Selected encoded payload would cross next member/directory')
            packed = request(tag+'-zip-deflate',payload_start,member['compressed_bytes_declared'])
            raw = expand_zip_member(packed,member['expanded_bytes_declared'],member['CRC32_declared_not_verified'])
            path = OUT / selected['preserved_filename']
            with path.open('xb') as output:
                output.write(raw)
            observations = inspect_encoded(raw)
            observations.update({'member_name':member['name'],'encoded_bytes':len(raw),'encoded_file_SHA256':sha(path),
                'actual_member_CRC32_verified':member['CRC32_declared_not_verified'], 'image_pixels_decoded_or_displayed':False,
                'matches_declared_JSON_width_height': [observations['width'],observations['height']] == policy['expected_JSON_dimensions'],
                'matches_expected_encoding': observations['format'] == selected['expected_encoding'],
                'GT_raster_background_unknown_class_mapping_or_physical_group_verified':False})
            write_json(OUT / (tag+'-header-observations.json'),observations)
            completed.append(observations)
            if not observations['matches_declared_JSON_width_height'] or not observations['matches_expected_encoding']:
                raise ValueError('Fixed encoded format/dimensions differ; no resize/rename/other-image fallback')
        stored = sum(p.stat().st_size for base in (OUT,RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes']-1048576
        result = {'UTC':utc(),'status':'TWO_ENCODED_TRAIN_HEADERS_COMPLETE','preregistered_HEAD':head,'policy_SHA256':sha(POLICY),
            'requests':requests,'actual_new_Range_body_bytes':received,'source_encoded_files_acquired':len(completed),
            'source_encoded_image_files_acquired':1,'source_encoded_mask_files_acquired':1,'header_observations':completed,
            'source_pixel_arrays_decoded_visual_QA_histogram_raster_features_scores_gradients_optimizer_CAL_TEST_API_CPP_ZIP_APK':0,
            'publisher_TEST_or_VAL_content_accessed':0,'old_metadata_HEAD_CD_JSON_jobs_or_fixtures_rerun':False,
            'full_archive_SHA_all_member_CRC_source_GT_roles_physical_groups_graph_or_minima_verified':False,
            'ready_for_pixel_or_training':False,'stored_bytes_before_final_report':stored,'disk_free':shutil.disk_usage(ROOT).free}
        write_json(OUT / 'encoded-headers-complete.json',result)
        print(json.dumps({'status':result['status'],'Range_bytes':received,'encoded_files':len(completed),
                          'formats':[r['format'] for r in completed],'dimensions':[[r['width'],r['height']] for r in completed],
                          'pixel_decode_optimizer':0}),flush=True)
    except BaseException as error:
        write_json(RUN / 'failure.json',{'UTC':utc(),'error':repr(error),'preregistered_HEAD':head,'requests':requests,
            'actual_Range_body_bytes':received,'completed_member_header_observations':completed,'originals_partials_headers_preserved':True,
            'image_pixel_decode_gradient_optimizer_CAL_TEST':0,'training_failure':False,
            'whole_stage_restart_retry_or_full_download_allowed':False,'ready_for_training':False})
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
