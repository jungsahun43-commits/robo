"""Bounded RAR5 header inspection, following RARLab's published format.

This module does not decompress or execute archived files. Cached Quick Open
headers must also match the original bytes before any selected extraction.
"""
from pathlib import PurePosixPath
import struct
import zlib

MAGIC = b'Rar!\x1a\x07\x01\x00'


def vint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('Truncated variable integer')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 2 ** 64:
                raise ValueError('Integer overflow')
            return value, position
    raise ValueError('Variable integer too long')


def encode_vint(value):
    if not 0 <= value < 2 ** 64:
        raise ValueError('Unsigned64 required')
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def header(data, position=0):
    start = position
    if position + 5 > len(data):
        raise ValueError('Truncated block')
    crc = struct.unpack_from('<I', data, position)[0]
    size, position = vint(data, position + 4)
    if size > 2 * 1024 ** 2 or position + size > len(data):
        raise ValueError('Header exceeds available bounded bytes')
    end = position + size
    if zlib.crc32(data[start + 4:end]) != crc:
        raise ValueError('Header CRC mismatch')
    kind, position = vint(data, position)
    flags, position = vint(data, position)
    extra, position = vint(data, position) if flags & 1 else (0, position)
    packed, position = vint(data, position) if flags & 2 else (0, position)
    if flags & (8 | 16):
        raise ValueError('Multivolume continuation unsupported')
    result = {'kind': kind, 'common_flags': flags, 'header_bytes': end - start,
              'packed': packed, 'extra_bytes': extra, 'CRC_verified': True}
    extra_start = end - extra
    if kind == 1:
        archive_flags, position = vint(data, position)
        if archive_flags & 7:
            raise ValueError('Only non-solid single volume accepted')
        result['archive_flags'] = archive_flags
    elif kind in (2, 3):
        file_flags, position = vint(data, position)
        unpacked, position = vint(data, position)
        attributes, position = vint(data, position)
        if file_flags & 8:
            raise ValueError('Unknown unpacked size unsupported')
        if file_flags & 2:
            position += 4
        if file_flags & 4:
            result['unpacked_CRC32'] = struct.unpack_from('<I', data, position)[0]
            position += 4
        compression, position = vint(data, position)
        host, position = vint(data, position)
        name_length, position = vint(data, position)
        if position + name_length > extra_start:
            raise ValueError('Filename overruns header')
        name = data[position:position + name_length].decode('utf-8')
        position += name_length
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or '\x00' in name:
            raise ValueError('Unsafe archive filename')
        if compression & 64:
            raise ValueError('Solid file dependency unsupported')
        result.update({'name': name, 'unpacked': unpacked, 'compression': compression,
                       'method': (compression >> 7) & 7, 'directory': bool(file_flags & 1),
                       'attributes': attributes, 'host': host})
    elif kind not in (5,):
        raise ValueError('Unsupported header type or encryption')
    if position > extra_start:
        raise ValueError('Fields overlap extra area')
    extras = []
    position = extra_start
    while position < end:
        length, position = vint(data, position)
        limit = position + length
        if limit > end:
            raise ValueError('Extra record out of bounds')
        record_type, position = vint(data, position)
        payload = data[position:limit]
        if kind in (2, 3) and record_type in (1, 5):
            raise ValueError('Encrypted files or redirection unsupported')
        extras.append((record_type, payload))
        position = limit
    return result, extras, end


def locator(data):
    if data[:8] != MAGIC:
        raise ValueError('Expected RAR5 signature')
    info, extras, _ = header(data, 8)
    if info['kind'] != 1:
        raise ValueError('Expected main header')
    for kind, payload in extras:
        if kind == 1:
            flags, position = vint(payload, 0)
            if flags & 1:
                offset, _ = vint(payload, position)
                if offset:
                    return 8 + offset
    raise ValueError('No usable Quick Open locator')


def quick_open_records(block, absolute_offset):
    service, _, position = header(block)
    if service.get('name') != 'QO' or service['kind'] != 3 or service['method'] != 0:
        raise ValueError('Expected uncompressed Quick Open service')
    end = position + service['packed']
    if end != len(block) or service['packed'] != service['unpacked']:
        raise ValueError('Quick Open length mismatch')
    if 'unpacked_CRC32' in service and zlib.crc32(block[position:end]) != service['unpacked_CRC32']:
        raise ValueError('Quick Open payload CRC mismatch')
    rows = []
    previous = -1
    while position < end:
        start = position
        crc = struct.unpack_from('<I', block, position)[0]
        size, position = vint(block, position + 4)
        limit = position + size
        if limit > end or size > 2 * 1024 ** 2 or zlib.crc32(block[start + 4:limit]) != crc:
            raise ValueError('Quick Open cache record CRC/length mismatch')
        flags, position = vint(block, position)
        relative, position = vint(block, position)
        data_size, position = vint(block, position)
        if flags != 0 or position + data_size != limit:
            raise ValueError('Unsupported Quick Open cache record')
        original_offset = absolute_offset - relative
        if not previous < original_offset < absolute_offset:
            raise ValueError('Cache offsets are not strictly ordered')
        cached = block[position:limit]
        info, _, consumed = header(cached)
        if consumed != len(cached):
            raise ValueError('Expected one complete cached header')
        rows.append({**info, 'offset': original_offset, 'header_hex': cached.hex(),
                     'data_offset': original_offset + len(cached)})
        previous = original_offset
        position = limit
    return rows


def single_member_archive(original_header, packed_payload):
    info, _, end = header(original_header)
    if info['kind'] != 2 or info['directory'] or end != len(original_header) or len(packed_payload) != info['packed']:
        raise ValueError('One non-solid regular member required')

    def block(body):
        raw = encode_vint(len(body)) + body
        return struct.pack('<I', zlib.crc32(raw)) + raw

    return MAGIC + block(b'\x01\x00\x00') + original_header + packed_payload + block(b'\x05\x00\x00')
