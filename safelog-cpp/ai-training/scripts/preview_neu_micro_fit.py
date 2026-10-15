"""Reserve source filename families before viewing three future FIT examples."""
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_neu_micro_candidate import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    assert not (DATA / 'QA-reservation.json').exists()
    manifest = read(DATA / 'candidate-manifest.json')
    assert not manifest['audit']['exact_RGB_or_any8orientation_pHash2_TEST_matches']
    rows = manifest['rows']
    training_families = ('NEU_Oil:Scratch', 'NEU_SDI:SDI', 'NEU_SPDI:SPDI')
    def family(row):
        return row['source_folder'] + ':' + re.sub(r'_\d+$', '', Path(row['image']).stem)
    selected = [next(r for r in rows if family(r) == f and Path(r['image']).stem.endswith('_1')) for f in training_families]
    reservation = {'UTC': datetime.now(timezone.utc).isoformat(), 'fixed_before_source_image_content_viewing': True,
        'candidate_manifest_sha256': sha(DATA / 'candidate-manifest.json'),
        'training_only_filename_families_after_visual_QA': list(training_families),
        'training_only_candidate_ids': [r['id'] for r in rows if family(r) in training_families],
        'unviewed_validation_candidate_ids': [r['id'] for r in rows if family(r) not in training_families],
        'visually_inspected_FIT_candidate_ids': [r['id'] for r in selected],
        'family_groups_are_conservative_filename_proxies_not_verified_physical_identity': True,
        'independent_field_validation': False, 'no_new_source_normal_photos': True,
        'no_source_whole_photo_FPR_or_new_app_category_claim': True,
        'full_training_protocol_fixed': False, 'new_model_scores_or_optimizer_updates': 0,
        'note': '41futureFIT positive-mask views/10unviewedVAL candidates, no source CAL or new independent TEST; add trusted normal FIT/VAL separately only in preregistered training study. Future split must respect all near-linked and viewed filename families.'}
    assert len(reservation['training_only_candidate_ids']) == 41
    assert len(reservation['unviewed_validation_candidate_ids']) == 10
    new(DATA / 'QA-reservation.json', reservation)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for line, row in enumerate(selected):
        assert sha(ROOT / row['image']) == row['image_sha256'] and sha(ROOT / row['mask']) == row['mask_sha256']
        with Image.open(ROOT / row['image']) as im: rgb = np.array(im.convert('RGB'))
        with Image.open(ROOT / row['mask']) as im: mask = np.array(im.convert('L')) > 0
        axes[line, 0].imshow(rgb)
        axes[line, 0].set_title(row['source_folder'] + ' / ' + Path(row['image']).stem)
        axes[line, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[line, 1].set_title('Published bright mask')
        axes[line, 2].imshow(rgb)
        axes[line, 2].imshow(np.ma.masked_where(~mask, mask), cmap='autumn', alpha=.5, vmin=0, vmax=1)
        axes[line, 2].set_title('Image and mask alignment')
        for ax in axes[line]: ax.axis('off')
    fig.tight_layout()
    target = DATA / 'reserved-FIT-3-QA.png'
    fig.savefig(target, dpi=120); plt.close(fig)
    print({'preview': str(target), 'image_sha256': sha(target), 'FIT_only_candidates': 41,
           'unviewed_VAL_candidates': 10, 'new_model_scores_or_optimizer': 0}, flush=True)


if __name__ == '__main__':
    main()
