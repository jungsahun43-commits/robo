"""FIT-only visual QA; publisher labels remain unchanged, no model or scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN = ROOT / 'runs/industrial-gear-FIT-review-v34'
MANIFEST = ROOT / 'data/industrial-gear-only-v28/manifest.json'
EXPECTED = 'fe7fce446e34002d47a24c0ad6b106468e8c128209739164d2e511705ae8779d'


def main():
    assert sha(MANIFEST) == EXPECTED and not RUN.exists()
    rows = {row['id']: row for row in read(MANIFEST)['rows']}
    audit = read(ROOT / 'reports/industrial-gear-selection-v32-diagnostic.json')
    pairs = [row for row in audit['development_near_opposite_label_pairs'] if row['phase'] == 'fit']
    assert len(pairs) == 5
    for pair in pairs:
        for key in ('left', 'right'):
            row = rows[pair[key]]
            assert row['phase'] == 'fit' and row['functional_component_category'] == 'gear'
            assert sha(ROOT / row['image']) == row['image_sha256']
    RUN.mkdir()
    # Freeze ids/labels/source paths before decoding any image pixels.
    plan = {'UTC': datetime.now(timezone.utc).isoformat(), 'stage': 'FIT_only_visual_QA_no_training',
            'manifest_sha256': EXPECTED, 'diagnostic_sha256': sha(ROOT / 'reports/industrial-gear-selection-v32-diagnostic.json'),
            'pairs': [{key: rows[pair[key]] for key in ('left', 'right')} for pair in pairs],
            'no_VAL_CAL_TEST_decoding': True, 'optimizer_or_new_model_scores': 0,
            'labels_unchanged': True, 'near_similarity_not_label_error': True,
            'review_scope': 'AI visual QA only; no manufacturer wear-tolerance expertise, no verified label correction'}
    new(RUN / 'plan-before-pixels.json', plan)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    canvas = Image.new('RGB', (740, 5 * 340), '#f4f4f4'); draw = ImageDraw.Draw(canvas)
    for index, pair in enumerate(pairs):
        for column, key in enumerate(('left', 'right')):
            row = rows[pair[key]]; x = column * 370 + 10; y = index * 340 + 8
            with Image.open(ROOT / row['image']) as original:
                assert original.size == (128, 128)
                view = original.convert('RGB').resize((256, 256), Image.Resampling.NEAREST)
            canvas.paste(view, (x, y + 64))
            draw.text((x, y), f"Pair {index+1}: {row['id']} | {row['label_name']}", font=font, fill='black')
            draw.text((x, y + 22), f"FIT group={row['group_id']} physical={row['physical_part_id']}", font=font, fill='black')
            draw.text((x, y + 42), f"acq={row['acq_id']}", font=font, fill='black')
    canvas.save(RUN / 'FIT-near-pairs.png')
    new(RUN / 'receipt.json', {'decoded_patch_inputs': 10, 'unique_FIT_patches': len({p[k] for p in pairs for k in ('left','right')}),
                             'new_model_scores': 0, 'optimizer': 0, 'image_sha256': sha(RUN / 'FIT-near-pairs.png'),
                             'original_manifest_SHA_unchanged': sha(MANIFEST) == EXPECTED})
    print({'sheet': str(RUN / 'FIT-near-pairs.png'), 'unique_FIT_patches': 8, 'no_labels_modified': True})


if __name__ == '__main__': main()
