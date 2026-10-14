"""Anonymous publisher download of labelled ball-screw wear crops; no code run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile
import requests
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/industrial-bsclass-v1'
API = 'https://radar.kit.edu/radar/dataset/bagit/download/xsvLWXhsaWvzMqkt'
MAX_BYTES = 1024*1024*1024


def main():
    assert not DATA.exists(); DATA.mkdir(parents=True)
    response = requests.get(API, timeout=30); response.raise_for_status(); url = response.json()['downloadUrl']
    archive = DATA/'publisher-bagit.archive'; total = 0; digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(20, 60)) as download:
        download.raise_for_status()
        length = int(download.headers.get('Content-Length') or 0)
        if length > MAX_BYTES: raise ValueError('Prespecified1GiB download budget exceeded')
        disposition = download.headers.get('Content-Disposition', '')
        with archive.open('xb') as out:
            for chunk in download.iter_content(1024*1024):
                total += len(chunk)
                if total > MAX_BYTES: raise ValueError('Prespecified1GiB download budget exceeded')
                out.write(chunk); digest.update(chunk)
                if total//(32*1024*1024) != (total-len(chunk))//(32*1024*1024): print(json.dumps({'downloaded_MiB': total//(1024*1024)}), flush=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            inventory = [{'path': n.filename, 'bytes': n.file_size, 'directory': n.is_dir()} for n in z.infolist()]; kind = 'zip'
    else:
        with tarfile.open(archive, 'r:*') as t:
            inventory = [{'path': n.name, 'bytes': n.size, 'directory': n.isdir()} for n in t]; kind = 'tar'
    receipt = {'status': 'publisher_archive_downloaded_not_yet_extracted_or_trained', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'primary_source': 'https://publikationen.bibliothek.kit.edu/1000133819', 'publisher_download_API': API,
        'archive_bytes': total, 'archive_sha256': digest.hexdigest(), 'archive_format': kind, 'content_disposition': disposition,
        'license': 'CC BY-NC4.0 per primary dataset page; preserve attribution', 'publisher_description_images': 21835,
        'publisher_description_normal': 11075, 'publisher_N_and_P_filename_labels': True, 'publisher_90degree_rotation_duplicates_need_grouping': True,
        'domain': 'ball_screw_drive_visible_surface_pitting_not_fastener_screw', 'inventory': inventory,
        'code_or_saved_models_extracted_or_executed': False, 'new_training_performed': False, 'new_performance_claimed': False}
    with (DATA/'download.json').open('x', encoding='utf-8', newline='\n') as out: json.dump(receipt, out, ensure_ascii=False, indent=2)
    print(json.dumps({'downloaded': 'BSClass_publisher', 'bytes': total, 'format': kind, 'inventory_entries': len(inventory), 'not_trained': True}), flush=True)


if __name__ == '__main__': main()
