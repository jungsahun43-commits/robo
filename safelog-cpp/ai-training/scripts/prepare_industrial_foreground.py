"""Audit sources and colour masks before fitting new foreground references."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.industrial_foreground import CATEGORIES, foreground_view, token_mask
from safelog_ai.belt_normal_roi import fingerprints, distance
from scripts.study_equipment_expansion import MANIFEST, read
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new

DATA = ROOT/'data/industrial-foreground-v12'
RUN = ROOT/'runs/industrial-foreground-v12'
REPORT = ROOT/'reports/industrial-foreground-v12-data-audit.json'


def source(row):
    assert sha(ROOT/row['image']) == row['image_sha256'], row['id']
    with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256'], row['id']
    for name, digest in zip(row['masks'], row['mask_sha256']): assert sha(ROOT/name) == digest
    return {**row, 'orientation_phashes': fingerprints(rgb), 'original_phase': row['phase']}


def main():
    assert not DATA.exists() and not RUN.exists()
    DATA.mkdir(); RUN.mkdir()
    selected = [r for r in read(MANIFEST)['rows'] if r['category'] in CATEGORIES and r['phase'] in ('fit', 'cal', 'test')]
    with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(source, selected))
    links = []; records = []
    for category in CATEGORIES:
        local = [r for r in rows if r['category'] == category]; roots = list(range(len(local)))
        def find(i):
            while roots[i] != i: roots[i] = roots[roots[i]]; i = roots[i]
            return i
        for i, a in enumerate(local):
            for j, b in enumerate(local[:i]):
                near = distance(a['orientation_phashes'], b['orientation_phashes']) <= 2
                if a['rgb_sha256'] == b['rgb_sha256'] or near:
                    x, y = find(i), find(j); roots[max(x, y)] = min(x, y)
                    links.append({'a': a['id'], 'b': b['id'], 'exact_RGB': a['rgb_sha256'] == b['rgb_sha256'], 'near_pHash': near})
        groups = {}
        for i, row in enumerate(local): groups.setdefault(find(i), []).append(row)
        for group in groups.values():
            group_id = min(r['id'] for r in group)
            for row in group: row['appearance_group_proxy'] = group_id
            test = [r for r in group if r['phase'] == 'test']
            cal = sorted((r for r in group if r['phase'] == 'cal'), key=lambda r: r['id'])
            if test:
                for row in group:
                    if row['phase'] != 'test': row['phase'] = 'excluded_orientation_near_TEST'
            elif cal:
                for row in group:
                    if row['phase'] == 'fit': row['phase'] = 'excluded_orientation_near_CAL'
                for row in cal[1:]: row['phase'] = 'excluded_duplicate_CAL_group'
            else:
                fit = sorted(group, key=lambda r: r['id'])
                for row in fit[1:]: row['phase'] = 'excluded_duplicate_FIT_group'
        fit_groups = {r['appearance_group_proxy'] for r in local if r['phase'] == 'fit'}
        cal_groups = {r['appearance_group_proxy'] for r in local if r['phase'] == 'cal'}
        test_groups = {r['appearance_group_proxy'] for r in local if r['phase'] == 'test'}
        assert not (fit_groups & cal_groups or fit_groups & test_groups or cal_groups & test_groups)
    # Mask quality exploration uses normal FIT only. CAL and TEST masks/scores
    # are not computed until the protocol has been committed.
    for row in rows:
        if row['phase'] != 'fit': continue
        assert row['label'] == 0 and row['publisher_split'] == 'train'
        with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
        try:
            _, mask, geometry = foreground_view(rgb)
            count = int(token_mask(mask[None], (32, 32)).sum())
            if count < 4: raise ValueError('Fewer than4 cells')
            row['foreground_FIT_audit'] = {'usable': True, 'fraction': geometry['foreground_fraction'], 'tokens': count, 'border_threshold': geometry['input_border_Lab_threshold']}
        except ValueError as error:
            row['foreground_FIT_audit'] = {'usable': False, 'reason': str(error)}
        records.append({'id': row['id'], **row['foreground_FIT_audit']})
    counts = {c: dict(Counter(r['phase'] for r in rows if r['category'] == c)) for c in CATEGORIES}
    summary = {}
    for category in CATEGORIES:
        fit = [r for r in rows if r['category'] == category and r['phase'] == 'fit']
        valid = [r for r in fit if r['foreground_FIT_audit']['usable']]
        summary[category] = {'FIT_photos': len(fit), 'usable_FIT': len(valid), 'usable_fraction': len(valid)/len(fit),
            'foreground_fraction_quantiles': np.quantile([r['foreground_FIT_audit']['fraction'] for r in valid], [0, .1, .5, .9, 1]).tolist() if valid else [],
            'tokens_quantiles': np.quantile([r['foreground_FIT_audit']['tokens'] for r in valid], [0, .1, .5, .9, 1]).tolist() if valid else []}
        indices = np.linspace(0, len(fit)-1, min(8, len(fit))).round().astype(int)
        sheet = Image.new('RGB', (4*256, 2*290), 'white'); draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(indices):
            row = fit[i]
            with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
            try:
                view, mask, _ = foreground_view(rgb)
                a = np.asarray(view).copy(); a[~mask.numpy()] = (np.asarray(view)[~mask.numpy()]*.2).astype(np.uint8)
                preview = Image.fromarray(a).resize((256, 256))
            except ValueError: preview = rgb.resize((256, 256))
            x, y = (j%4)*256, (j//4)*290; sheet.paste(preview, (x, y))
            draw.text((x+3, y+260), Path(row['id']).name+' '+str(row['foreground_FIT_audit']['usable']), fill='black')
        sheet.save(RUN/(category+'-normalFIT-mask-review.png'))
    audit = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': sha(MANIFEST), 'counts': counts,
        'mask_prototype_sha256': sha(ROOT/'safelog_ai/industrial_foreground.py'), 'normal_FIT_mask_review': summary,
        'appearance_near_group_links': links, 'source_images_and_masks_SHA_rechecked': len(rows),
        'exact_and_any8orientation_pHash2_groups_disjoint': True, 'physical_component_or_acquisition_groups_verified': False,
        'publisher_TEST_or_duplicates_in_FIT_CAL': False, 'CAL_TEST_masks_or_model_scores_computed': False,
        'new_optimizer_updates': 0, 'colour_mask_is_verified_part_ground_truth': False, 'independent_field_validated': False}
    new(DATA/'manifest.json', {'rows': rows, 'audit': audit})
    new(REPORT, audit)
    new(RUN/'normalFIT-mask-audit.json', {'summary': summary, 'rows': records})
    print({'counts': counts, 'FIT_masks': summary, 'near_links': len(links)}, flush=True)


if __name__ == '__main__': main()
