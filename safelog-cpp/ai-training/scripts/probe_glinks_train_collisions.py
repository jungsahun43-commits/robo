"""TRAIN-only pHash collision investigation; never changes split or trains."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.image_pair_probe import orientation_rgb_probe
from safelog_ai.belt_normal_roi import distance, fingerprints
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

DATA = ROOT / 'data/industrial-glinks-source-v35'
RUN = ROOT / 'runs/industrial-glinks-collision-probe-v36'
POLICY = ROOT / 'reports/industrial-glinks-collision-v36-policy.json'
SOURCES = ('scripts/probe_glinks_train_collisions.py', 'safelog_ai/image_pair_probe.py', 'tests/test_image_pair_probe.py')


def freeze():
    assert not POLICY.exists() and not RUN.exists()
    rows = read(DATA / 'manifest.json')['rows']
    uuids = sorted({row['acquisition_uuid_proxy'] for row in rows if row['publisher_role'] == 'train'},
                   key=lambda value: hashlib.sha256(('glinks36:' + value).encode()).hexdigest())[:2]
    pairs = [pair for pair in read(DATA / 'orientation-near-pairs.json')
             if all(rows[i]['publisher_role'] == 'train' for i in pair)
             and {rows[i]['acquisition_uuid_proxy'] for i in pair} == set(uuids)]
    pairs = sorted(pairs)[:8]
    assert len(pairs) == 8
    ids = sorted({index for pair in pairs for index in pair})
    RUN.mkdir()
    plan = {'UTC': datetime.now(timezone.utc).isoformat(), 'fixed_before_new_RGB_comparison_or_viewing': True,
            'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
            'manifest_sha256': sha(DATA / 'manifest.json'), 'near_pairs_sha256': sha(DATA / 'orientation-near-pairs.json'),
            'permanently_development_only_uuid_proxies_not_future_VAL_CAL_TEST': uuids,
            'row_indices': ids, 'pair_indices': pairs,
            'selection': 'two publisherTRAIN UUIDs by fixedSHA(glinks36:+UUID), first8 lexicographic pHash-near cross-UUID pairs. No TEST pictures or model scores select examples.',
            'comparison': 'unaltered sourceRGB read then fixed256square bilinear diagnostic thumbnail; minMAD over8orientations, report sameorientationRMSE/p95. No foreground mask or selected ROI.',
            'positive_controls': 'first selected original thumbnail: exact dihedral duplicate and JPEGquality90 roundtrip. Known edited-photo controls only, not newly collected natural images.',
            'visual_review': 'first2 fixed pairs, four TRAINinput views, thumbnails marked originalpublisher labels; AI QA only',
            'do_not_change': 'v35 failure/manifest/exclusions/pHashthreshold2/publisherTEST/GT remain. No duplicate confirmation threshold or new training split chosen here.',
            'features_gradients_optimizer_CAL_TEST_model_scores': 0, 'new_units_passed': 2,
            'next_rule_requires_separate_protocol': True,
            'limitation': 'UUID is acquisition proxy, not verified distinct physical objects; RGB disagreement alone does not prove independence or a label error.'}
    new(POLICY, plan);print({'selected_TRAIN_pairs': len(pairs), 'unique_TRAIN_photos': len(ids), 'permanently_development_only_UUIDs': uuids})


def probe():
    policy = read(POLICY)
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
    for name, expected in policy['source_sha256'].items():
        assert sha(ROOT / name) == expected
        assert hashlib.sha256(subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])).hexdigest() == expected
    assert sha(DATA / 'manifest.json') == policy['manifest_sha256']
    assert sha(DATA / 'orientation-near-pairs.json') == policy['near_pairs_sha256']
    assert not (RUN / 'result.json').exists()
    rows = read(DATA / 'manifest.json')['rows']; thumbnails = {}
    for index in policy['row_indices']:
        row = rows[index];assert row['publisher_role'] == 'train' and row['label'] == 0
        assert row['acquisition_uuid_proxy'] in policy['permanently_development_only_uuid_proxies_not_future_VAL_CAL_TEST']
        assert sha(ROOT / row['image']) == row['image_sha256']
        with Image.open(ROOT / row['image']) as image:
            thumbnails[index] = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    results = []
    for left, right in policy['pair_indices']:
        results.append({'left': rows[left]['id'], 'right': rows[right]['id'],
                        'original_pHash8_min_distance': distance(rows[left]['orientation_phashes'], rows[right]['orientation_phashes']),
                        **orientation_rgb_probe(thumbnails[left], thumbnails[right])})
    first = thumbnails[policy['row_indices'][0]]
    encoded = io.BytesIO();first.save(encoded, format='JPEG', quality=90);encoded.seek(0)
    with Image.open(encoded) as image: jpeg = image.convert('RGB')
    controls = {'dihedral': orientation_rgb_probe(first, first.transpose(Image.Transpose.TRANSVERSE)),
                'JPEG90': {**orientation_rgb_probe(first, jpeg), 'pHash8_distance': distance(fingerprints(first), fingerprints(jpeg))}}
    assert controls['dihedral']['mean_absolute_RGB_difference'] == 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    sheet = Image.new('RGB', (600, 640), '#f4f4f4');draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(policy['pair_indices'][:2]):
        for column, row_index in enumerate(pair):
            row = rows[row_index];x = column * 300 + 8;y = index * 320 + 4
            draw.text((x, y), 'Publisher TRAIN / good / QA only', font=font, fill='black')
            draw.text((x, y + 20), Path(row['member']).name[:20], font=font, fill='black')
            sheet.paste(thumbnails[row_index], (x, y + 44))
    sheet.save(RUN / 'TRAIN-pHash-pairs.png')
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'policy_sha256': sha(POLICY), 'pairs': results,
              'known_same_image_controls': controls, 'unique_decoded_TRAIN_photos': len(thumbnails),
              'TEST_VAL_CAL_photos_viewed': 0, 'model_features_gradients_optimizer_or_scores': 0,
              'v35_manifest_unchanged': sha(DATA / 'manifest.json') == policy['manifest_sha256'],
              'new_training_split_or_duplicate_acceptance_threshold': False, 'new_physical_identity_proof': False,
              'contact_sheet_sha256': sha(RUN / 'TRAIN-pHash-pairs.png')}
    new(RUN / 'result.json', result)
    new(ROOT / 'reports/industrial-glinks-collision-v36-result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('stage', choices=('freeze', 'probe'))
    args = parser.parse_args();globals()[args.stage]()
