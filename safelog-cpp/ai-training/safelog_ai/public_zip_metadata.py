"""Strict bounded ZIP directory metadata, without extraction or pixel decoding."""
from pathlib import PurePosixPath
import stat
import struct


def check_range(status, headers, start, count, total, etag):
    if status != 206 or headers.get('Content-Range') != f'bytes {start}-{start+count-1}/{total}':
        raise ValueError('Exact public 206 range required; never read full fallback')
    if headers.get('ETag') != etag or headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('Archive identity/encoding changed')
    if headers.get('Content-Length') != str(count):
        raise ValueError('Range length not exact')


def end_directory(tail, total, max_directory_bytes=8*1024**2):
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0 or len(tail)-pos < 22:
        raise ValueError('Missing or truncated ZIP end record')
    _, disk, cd_disk, disk_entries, entries, size, offset, comment = struct.unpack_from('<4s4H2IH', tail, pos)
    if disk or cd_disk or disk_entries != entries or entries == 65535 or size == 0xffffffff or offset == 0xffffffff:
        raise ValueError('Multi-disk/ZIP64 unsupported in this policy')
    if pos+22+comment != len(tail) or not 0 < size <= max_directory_bytes or not 0 < entries <= 50000:
        raise ValueError('Invalid ZIP directory budget/end')
    if offset+size != total-len(tail)+pos:
        raise ValueError('Directory does not directly precede end record')
    return {'entries': entries, 'bytes': size, 'offset': offset}


def parse_directory(data, expected_entries, archive_bytes):
    rows, names, pos = [], set(), 0
    while pos < len(data):
        if len(data)-pos < 46:
            raise ValueError('Truncated ZIP directory header')
        f = struct.unpack_from('<4s6H3I5H2I', data, pos)
        if f[0] != b'PK\x01\x02':
            raise ValueError('Unknown directory record')
        flags, method, crc, packed, size = f[3], f[4], f[7], f[8], f[9]
        n, extra, comment, disk, attrs, offset = f[10], f[11], f[12], f[13], f[15], f[16]
        stop = pos+46+n+extra+comment
        if stop > len(data) or disk or flags & (1 | 64) or method not in (0, 8):
            raise ValueError('Unsafe/truncated/encrypted/unsupported ZIP member')
        name = data[pos+46:pos+46+n].decode('utf-8' if flags & 0x800 else 'cp437')
        path = PurePosixPath(name)
        if not name or path.is_absolute() or path.as_posix() != name.rstrip('/') or '..' in path.parts or any(c in name for c in ('\\', ':', '\x00')) or name in names or stat.S_ISLNK(attrs >> 16):
            raise ValueError('Unsafe or duplicate ZIP path')
        if max(packed, size, offset) >= 0xffffffff or offset+30+n+packed > archive_bytes:
            raise ValueError('ZIP64 or invalid member bounds')
        names.add(name)
        rows.append({'member': name, 'bytes': size, 'packed': packed, 'method': method,
                     'flags': flags, 'CRC32_declared': f'{crc:08x}', 'local_header_offset': offset,
                     'directory': name.endswith('/')})
        pos = stop
    if len(rows) != expected_entries:
        raise ValueError('Directory member count changed')
    return rows
