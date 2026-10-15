"""Public author-linked folder metadata only; no images or browser credentials."""
from collections import Counter
from datetime import datetime, timezone
import codecs
import hashlib
import json
from pathlib import Path
import re
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha

OUT = ROOT / 'data/industrial-mpdd2-source-v37'
ROOT_FOLDER = '1KAJfPPuusS4fEbPocHtIjLehCkpSEUSS'
FOLDER_MIME = 'application/vnd.google-apps.folder'


def public_entries(html):
    match = re.search(r"window\['_DRIVE_ivd'\] = '((?:\\.|[^'])*)';", html)
    if match is None: raise ValueError('No public listing; do not log in or bypass access')
    payload = match[1].replace(r'\/', '/')
    if not payload.isascii() or len(payload) > 16 * 1024 ** 2:
        raise ValueError('Unexpected encoded listing')
    # Decode only a data string and JSON. Never evaluate JavaScript or source code.
    value = json.loads(codecs.decode(payload, 'unicode_escape'))
    rows = value[0] or []
    if not isinstance(rows, list): raise ValueError('Unexpected public listing structure')
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 4: raise ValueError('Unexpected public item')
        key, _, name, mime = row[:4]
        if not re.fullmatch(r'[A-Za-z0-9_-]{10,100}', key): raise ValueError('Unexpected public item id')
        if not isinstance(name, str) or not isinstance(mime, str): raise ValueError('Unexpected name or mime')
        result.append({'id': key, 'name': name, 'mime': mime, 'raw_metadata': row})
    return result


def main():
    assert not OUT.exists()
    OUT.mkdir()
    new(OUT / 'intent-before-network.json', {
        'UTC': datetime.now(timezone.utc).isoformat(), 'source': 'author MPDD2 README public Drive link',
        'license_source': 'https://github.com/stepanje/MPDD2/blob/main/LICENSE',
        'license': 'CC BY-NC-SA4.0, DOI10.1109/ICUMT57764.2022.9943437',
        'selected_metadata_only_categories': ['bracket_grey', 'rosette'],
        'original_TEST_and_ground_truth_reserved_not_TRAIN': True,
        'no_image_download_decoding_viewing_or_model': True,
        'root_README_SHA256': sha(ROOT / 'data/industrial-source-candidates-v35/mpdd2-author-readme.txt'),
        'full_training_protocol_fixed': False})
    receipts = []; discovered = []; visited = set()
    def crawl(key, path, depth=0):
        if depth > 6 or len(visited) >= 32 or key in visited: raise ValueError('Unexpected tree bounds')
        visited.add(key)
        url = 'https://drive.google.com/drive/folders/' + key
        response = requests.get(url, timeout=(20, 40));response.raise_for_status()
        if not response.url.startswith('https://drive.google.com/'): raise ValueError('Nonpublic redirect; stop')
        raw = response.content
        if len(raw) > 16 * 1024 ** 2: raise ValueError('Unexpected metadata size')
        target = OUT / f'public-folder-{len(receipts):02d}.html'
        with target.open('xb') as output: output.write(raw)
        entries = public_entries(response.text)
        receipts.append({'id': key, 'path': path, 'url': url, 'status': response.status_code,
                         'response_bytes': len(raw), 'response_sha256': hashlib.sha256(raw).hexdigest(), 'entries': len(entries)})
        for entry in entries:
            entry_path = path + '/' + entry['name'] if path else entry['name']
            record = {**entry, 'relative_path': entry_path, 'parent_id': key}
            discovered.append(record)
            if entry['mime'] == FOLDER_MIME and (path or entry['name'] in ('bracket_grey', 'rosette')):
                crawl(entry['id'], entry_path, depth + 1)
        print({'metadata_folder': path or 'root', 'entries': len(entries)}, flush=True)
    try:
        crawl(ROOT_FOLDER, '')
    except BaseException as error:
        new(OUT / 'failure.json', {'error': repr(error), 'receipts': receipts, 'discovered': discovered,
                                  'images_downloaded': 0, 'optimizer': 0})
        raise
    files = [row for row in discovered if row['mime'] != FOLDER_MIME]
    inventory = {'UTC': datetime.now(timezone.utc).isoformat(), 'receipts': receipts, 'items': discovered,
                 'images_downloaded_decoded_viewed': 0, 'model_features_gradients_optimizer': 0,
                 'credentials_or_personal_information_submitted': False,
                 'listed_files': len(files), 'paths_by_parent': dict(Counter(row['relative_path'].rsplit('/', 1)[0] for row in files)),
                 'all_files_visible_in_public_listing_not_guaranteed': True,
                 'metadata_not_verified_new_nonduplicate_photographs': True,
                 'original_TEST_and_ground_truth_reserved_not_training': True}
    new(OUT / 'public-inventory.json', inventory)
    summary = {key: value for key, value in inventory.items() if key not in ('receipts', 'items')}
    summary['source_inventory_sha256'] = sha(OUT / 'public-inventory.json')
    new(ROOT / 'reports/industrial-mpdd2-v37-public-metadata.json', summary)
    print(summary, flush=True)


if __name__ == '__main__': main()
