"""Extract more author-labeled TRAIN/VAL frames from verified local bytes only.

Creates independent sparse RAR volumes; never edits previous inputs or requests
network data. Selection is fixed before decoding annotations or model scores.
"""
import base64
import hashlib
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_beltcrack_starter import rasterize, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/equipment-beltcrack'
INDEX = SOURCE / 'range-index'
OLD = SOURCE / 'starter-available-v1'
DATA = SOURCE / 'expanded-cached-v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)


def byte_source(row, entry):
    start = row['offset']
    end = start + row['header_size'] + row['packed'] - 1
    cache = INDEX / f'selected-{entry["fileName"]}-{start}-{end}.bin'
    partial = SOURCE / 'archives' / (entry['fileName'] + '.partial')
    if cache.is_file() and cache.stat().st_size == end-start+1:
        return cache, 0
    if partial.is_file() and partial.stat().st_size > end:
        return partial, start
    return None


def selection(members, entries, previous):
    """Union previous sample with evenly spaced frames having all packed parts."""
    parts, groups = defaultdict(list), defaultdict(list)
    pattern = re.compile(r'BeltCrack14ks/json/(train|val)/(data\d+)/(\d+)\.json')
    for row in members:
        if pattern.fullmatch(row['name']):
            parts[row['name']].append(row)
    available = {}
    for name, rows in parts.items():
        if all(byte_source(row, entries[row['volume']]) for row in rows):
            split, sequence, number = pattern.fullmatch(name).groups()
            item = {'name': name, 'split': split, 'sequence': sequence}
            available[name] = item
            groups[split, sequence].append((int(number), name))
    chosen = {row['name']: row for row in previous}
    assert set(chosen) <= set(available)
    for (split, sequence), group in sorted(groups.items()):
        ordered = [name for _, name in sorted(group)]
        maximum = 96 if split == 'train' else 64
        indices = np.unique(np.linspace(0, len(ordered)-1, min(maximum, len(ordered)), dtype=int))
        for index in indices:
            chosen[ordered[index]] = available[ordered[index]]
    return sorted(chosen.values(), key=lambda r: (r['split'], int(r['sequence'][4:]), int(Path(r['name']).stem))), parts, {
        split+'/'+group: len(rows) for (split, group), rows in sorted(groups.items())}


def acquire():
    assert not DATA.exists(), 'Do not overwrite an earlier extraction'
    members = read(INDEX / 'members.json')['members']
    entries = read(SOURCE / 'source-files.json')['data']
    previous = read(OLD / 'selection.json')['selected']
    chosen, parts, available_counts = selection(members, entries, previous)
    new(DATA / 'selection.json', {'selected': chosen,
        'rule': 'Previous679 union numerical evenly spaced complete locally cached TRAIN96/VAL64 per sequence; before new GT or scores',
        'available_original_json_counts': available_counts,
        'source_index_sha256': sha(INDEX / 'members.json'),
        'prior_selection_sha256': sha(OLD / 'selection.json'),
        'network_requests': 0, 'source_test_selected': False})
    sparse = DATA / 'sparse-extraction-only'
    sparse.mkdir()
    # Preserve the original sparse volumes. Copy validated header blocks into a
    # separate sparse workspace; incomplete sparse RARs are not full downloads.
    for entry in entries:
        target = sparse / entry['fileName']
        target.touch(exist_ok=False)
        subprocess.run(['fsutil', 'sparse', 'setflag', str(target)], check=True, capture_output=True)
        with target.open('r+b') as out:
            out.truncate(entry['size'])
            for block in INDEX.glob(entry['fileName'] + '-*.bin'):
                match = re.fullmatch(re.escape(entry['fileName']) + r'-(\d+)-(\d+)\.bin', block.name)
                assert match
                start, end = map(int, match.groups())
                content = block.read_bytes()
                assert len(content) == end-start+1
                out.seek(start)
                out.write(content)
    for volume, entry in enumerate(entries):
        with (INDEX / entry['fileName']).open('rb') as original, (sparse / entry['fileName']).open('r+b') as out:
            for row in members:
                if row['volume'] != volume:
                    continue
                original.seek(row['offset'])
                header = original.read(row['header_size'])
                assert len(header) == row['header_size']
                out.seek(row['offset'])
                out.write(header)
    ranges = []
    for i, chosen_row in enumerate(chosen, 1):
        for row in parts[chosen_row['name']]:
            entry = entries[row['volume']]
            origin = byte_source(row, entry)
            assert origin
            source, offset = origin
            length = row['header_size'] + row['packed']
            with source.open('rb') as handle:
                handle.seek(offset)
                content = handle.read(length)
            with (INDEX / entry['fileName']).open('rb') as handle:
                handle.seek(row['offset'])
                expected = handle.read(row['header_size'])
            assert len(content) == length and content[:len(expected)] == expected
            with (sparse / entry['fileName']).open('r+b') as handle:
                handle.seek(row['offset'])
                handle.write(content)
            ranges.append({'name': row['name'], 'volume': row['volume'],
                'offset': row['offset'], 'bytes': length,
                'sha256': hashlib.sha256(content).hexdigest(), 'actual_header_verified': True})
        if i % 100 == 0:
            print(json.dumps({'packed_members_verified': i, 'selected': len(chosen)}), flush=True)
    listing = DATA / 'selected-members.txt'
    listing.write_text('\n'.join(row['name'] for row in chosen), encoding='utf-8')
    destination = DATA / 'original-annotations'
    destination.mkdir()
    command = [str(ROOT / 'runs/equipment-tools/7zip-26.04/7z.exe'), 'x',
        str(sparse / entries[0]['fileName']), '-i@'+str(listing), '-pcv205',
        '-o'+str(destination), '-aos', '-spd', '-scsUTF-8']
    process = subprocess.run(command, capture_output=True)
    (DATA / 'extraction.log').write_bytes(process.stdout+process.stderr)
    assert process.returncode == 0, (process.stdout+process.stderr).decode(errors='replace')[-2000:]
    acquired = []
    for row in chosen:
        path = destination.joinpath(*row['name'].split('/'))
        assert path.resolve().is_relative_to(destination.resolve()) and path.is_file()
        acquired.append({**row, 'sha256': sha(path), 'bytes': path.stat().st_size})
    new(DATA / 'acquisition.json', {'status': 'complete', 'members': acquired,
        'source': 'https://doi.org/10.57760/sciencedb.31181', 'license': 'CC-BY-SA-4.0',
        'ranges': ranges, 'network_requests': 0, 'source_test_photos_or_labels_extracted': 0,
        'original_headers_verified': True, 'extractor_member_checksums_passed': True,
        'whole_archive_md5_verified': False, 'sparse_volumes_are_complete_archives': False,
        'script_sha256': sha(Path(__file__))})
    print(json.dumps({'acquired': len(acquired), 'by_split': dict(Counter(r['split'] for r in acquired))}), flush=True)


def prepare():
    assert not (DATA / 'manifest.json').exists()
    acquisition = read(DATA / 'acquisition.json')
    assert acquisition['extractor_member_checksums_passed']
    prepared = DATA / 'prepared'
    prepared.mkdir()
    rows, pixel_ids = [], {}
    for i, entry in enumerate(acquisition['members']):
        path = DATA / 'original-annotations' / entry['name']
        assert sha(path) == entry['sha256']
        annotation = read(path)
        raw = base64.b64decode(annotation['imageData'], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGB')
            pixels = np.asarray(image)
            pixel_id = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            assert pixel_id not in pixel_ids, 'Duplicate embedded RGB: '+entry['name']
            pixel_ids[pixel_id] = entry['name']
            frame, mask, valid = rasterize(annotation, image)
        # Empty shape lists would not establish whole-photo mechanical health.
        # This experiment refuses them rather than guessing normal ground truth.
        assert annotation['shapes'], 'Empty annotation: normal status needs a separate author-label audit'
        item = {'id': entry['name'], 'split': entry['split'], 'sequence': entry['sequence'],
            'annotation_sha256': entry['sha256'], 'embedded_image_sha256': hashlib.sha256(raw).hexdigest(),
            'decoded_rgb_sha256': pixel_id, 'width': annotation['imageWidth'], 'height': annotation['imageHeight'],
            'publisher_crack_shapes': len(annotation['shapes']), 'photo_positive': True,
            'positive_pixels': int((np.asarray(mask)>0).sum()), 'valid_pixels': int((np.asarray(valid)>0).sum())}
        assert item['positive_pixels'] > 0, 'Positive polygon vanished at640'
        for kind, value in [('image', frame), ('mask', mask), ('valid', valid)]:
            dest = prepared / f'{i}-{kind}.png'
            value.save(dest)
            item[kind] = dest.relative_to(DATA).as_posix()
            item[kind+'_sha256'] = sha(dest)
        rows.append(item)
        if i % 100 == 0:
            print(json.dumps({'prepared_photos': i}), flush=True)
    old = read(OLD / 'manifest.json')
    by_id = {r['id']: r for r in rows}
    for prior in old['rows']:
        for key in ('split', 'sequence', 'decoded_rgb_sha256', 'image_sha256', 'mask_sha256', 'valid_sha256'):
            assert by_id[prior['id']][key] == prior[key], 'Baseline input changed: '+prior['id']
    groups = {s: sorted({r['sequence'] for r in rows if r['split']==s}) for s in ('train','val')}
    assert not set(groups['train']) & set(groups['val'])
    new(DATA / 'manifest.json', {'status': 'frozen', 'rows': rows, 'input_size': 640,
        'counts': dict(Counter(r['split'] for r in rows)), 'sequences': groups,
        'normal_photo_labels': 0, 'decoded_exact_duplicates': 0,
        'baseline679_rgb_and_prepared_inputs_identical': True,
        'source_test_photos_or_labels_used': False, 'normal_photo_fpr_measurable': False,
        'photo_source': old['photo_source'], 'label_source': old['label_source'],
        'pixel_truth': old['pixel_truth'], 'padding_excluded_from_loss_and_metrics': True,
        'selection_sha256': sha(DATA / 'selection.json'), 'acquisition_sha256': sha(DATA / 'acquisition.json'),
        'preparation_script_sha256': sha(Path(__file__))})
    print(json.dumps({'prepared': len(rows), 'counts': dict(Counter(r['split'] for r in rows)), 'normal': 0}), flush=True)


if __name__ == '__main__':
    acquire()
    prepare()
