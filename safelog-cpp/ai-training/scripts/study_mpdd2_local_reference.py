"""One fixed local-reference experiment, not another optimizer training claim."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import study_mpdd2_normal_context as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from safelog_ai.mpdd2_local_reference import MPDD2ReferenceMain, VERSION, white_local_features, reference_evidence
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.research_delta import pack_delta, restore_delta
from safelog_ai.retention_distillation import state_sha256

RUN = ROOT / 'runs/industrial-mpdd2-local-reference-v41'
PROTOCOL = ROOT / 'reports/industrial-mpdd2-local-reference-v41-protocol.json'
SOURCES = ('safelog_ai/mpdd2_local_reference.py', 'scripts/study_mpdd2_local_reference.py', 'tests/test_mpdd2_local_reference.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    assert read(prior.RUN / 'verification.json')['parent_SHA256_unchanged']
    assert not read(prior.RUN / 'evaluation.json')['eligible_research_candidate']
    assert sha(prior.START) == prior.EXPECTED and shutil.disk_usage(ROOT).free > 1024 ** 3
    assert len(prior.rows_for('fit')) == 113
    RUN.mkdir()
    protected = read(prior.RUN / 'protected-stat.json')
    for path in prior.RUN.rglob('*'):
        if path.is_file() and not path.name.endswith(('.lock', '.log')):
            stat = path.stat()
            protected[path.relative_to(ROOT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    new(RUN / 'protected-stat.json', protected)
    policy = {'UTC': utc(), 'schema': VERSION, 'fixed_before_new_local_features_bank_or_scores': True,
        'source_SHA256': {name: sha(ROOT / name) for name in SOURCES},
        'legacy_Python_SHA256': {path.relative_to(ROOT).as_posix(): sha(path) for directory in ('safelog_ai', 'scripts', 'tests')
                               for path in (ROOT / directory).glob('*.py') if path.relative_to(ROOT).as_posix() not in SOURCES},
        'parent_SHA256': prior.EXPECTED, 'manifest_SHA256': sha(prior.MANIFEST),
        'protected_stat_SHA256': sha(RUN / 'protected-stat.json'), 'v40_evaluation_SHA256': sha(prior.RUN / 'evaluation.json'),
        'hypothesis': 'coarse4x4 context misses local differences; fixed normal patch memory on finer shared features is a different non-gradient estimator. This does not prove cause of v40 misses.',
        'data': 'unchanged v39 FIT113/VAL26/CAL25/TEST49normal91anomalous;31nearTEST TRAIN excluded; no new labels or photos',
        'input': 'original RGB256x256 ImageNet normalized, no crop/rotation/augmentation/foreground selection; same geometry for all phases',
        'features': 'one accepted encoder forward_with_fine stage3fine24@64x64 and low40@32x32, bilinearlow to64; separately unitnormalize then concatenate,3x3avg stride1 pad1 count_include_padFalse,constant.01backgroundtoken,unitnormalize65,FP16 roundtrip cache/runtime',
        'bank': 'only original FIT113; fixed spatial positions row,col=2,6,...62 (16x16/photo), all28928vectors in rowID order; no CAL/TEST, score-based coreset or learned projection',
        'model': 'acceptedv16 all tensors/buffers/metadata frozen; added buffer mpdd2_white_reference only. No separate encoder, category embedding, gradient or optimizer.',
        'optimizer': 0, 'new_features_not_new_natural_photos': True,
        'cache_contract': 'phase cache identities bind rawSHA/RGBSHA/sourceID/encoderSHA/protocolSHA/order/geometry/dtype; low/high vs normalforward bitwise check firstFIT only, original state hash before/after',
        'disk': 'all335 localfeatures FP16~179MB plus<4MBbank, <5MBdelta and~36MBstatledger; reserve1GiB, do not delete previous originals/cache/failed models',
        'photo_score': 'nearestcosine normalbank per64x64query, chunks1024/8192, mean largest41of4096 distances; rawscore not failureprobability',
        'VAL': 'normalVAL26 scoring diagnostic only, no epoch/architecture/threshold choice or bank addition',
        'CAL': 'normalCAL25 rankceil26*.85=23 (alpha.15), strict score>cut, store before one TEST feature/score phase',
        'eligibility': 'new normalbank all113FIT registered; TESTeachlabel>=20/each>=3conservativegroups, photoFPandFNboth<=.20, parentallstate/meta exact. This estimator has no optimizer/selectedepoch; not a relaxation of v40 neural epoch guard.',
        'source_scope': 'MPDD2 white-bracket aligned256closeups only, not old MPDD white, general machinery, internalhealth or smartphone field distribution',
        'no_position_ground_truth': True, 'existing_primary_maps_pixel_thresholds_unchanged': True,
        'prior_public_TEST_seen_in_v40': True, 'independent_field_validation': False, 'physical_capture_groups_unverified': True,
        'target_transfer_evaluated': False, 'no_auto_app_adoption': True,
        'adoption_requires': 'fresh CPU/GPU/realC++/allalias/oldlocationPNG tests and strict source-domain API before any new ZIP release',
        'new_units_passed': 3}
    new(PROTOCOL, policy)
    print({'protocol_SHA256': sha(PROTOCOL), 'normal_FIT_registration_photos': 113, 'planned_reference_vectors': 28928, 'optimizer': 0}, flush=True)


def validate():
    policy = read(PROTOCOL)
    for name, expected in policy['source_SHA256'].items():
        assert sha(ROOT / name) == expected
    assert sha(prior.START) == prior.EXPECTED and sha(prior.MANIFEST) == policy['manifest_SHA256']
    assert sha(RUN / 'protected-stat.json') == policy['protected_stat_SHA256']
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory=' + ROOT.parents[1].as_posix(), '-C', str(ROOT.parents[1])]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git + ['show', 'HEAD:safelog-cpp/ai-training/' + name])).hexdigest() == sha(ROOT / name)
    return policy


def run():
    policy = validate()
    assert not (RUN / 'evaluation.json').exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = RUN / 'operation.lock'
    assert not lock.exists()
    new(lock, {'pid': os.getpid(), 'UTC': utc(), 'operation': 'normal_reference_registration_and_fixed_evaluation'})
    try:
        parent = torch.load(prior.START, map_location='cpu', weights_only=True)
        model = MPDD2ReferenceMain(parent['reference_counts'], parent['native_specs'], parent['wear_variant'], parent['surface_reference_specs'])
        missing, extra = model.load_state_dict(parent['state_dict'], strict=False)
        assert not extra and missing == ['mpdd2_white_reference']
        model.eval().requires_grad_(False).cuda()
        before = state_sha256({name: tensor for name, tensor in model.state_dict().items() if name in parent['state_dict']})
        assert before == state_sha256(parent['state_dict'])

        def phase_features(phase):
            path = RUN / (phase + '-features.pt')
            assert not path.exists(), 'Do not repeat a partial reference run without audited recovery'
            rows = prior.rows_for(phase)
            targets = []
            with torch.inference_mode():
                for index, row in enumerate(rows):
                    assert sha(ROOT / row['image']) == row['image_sha256']
                    with Image.open(ROOT / row['image']) as image:
                        rgb = image.convert('RGB')
                    assert rgb.size == (256, 256) and hashlib.sha256(rgb.tobytes()).hexdigest() == row['rgb_sha256']
                    x = normalized(rgb)[None].cuda()
                    values = forward_with_fine(model.backbone, x)
                    if phase == 'fit' and index == 0:
                        original = model.backbone(x)
                        assert torch.equal(original['low'], values['low']) and torch.equal(original['high'], values['high'])
                    targets.append(white_local_features(values).cpu().half()[0])
                    if index % 40 == 0:
                        print({'local_features': phase, 'photos': index + 1, 'total': len(rows)}, flush=True)
            features = torch.stack(targets)
            with path.open('xb') as output:
                torch.save({'targets': features, 'protocol_SHA256': sha(PROTOCOL), 'parent_SHA256': prior.EXPECTED,
                    'identities': [{key: row[key] for key in ('id', 'image_sha256', 'rgb_sha256')} for row in rows],
                    'geometry': [256, 256, 65, 64, 64], 'dtype': 'FP16-roundtrip'}, output)
            return rows, features

        _, fit = phase_features('fit')
        bank = fit[:, :, 2::4, 2::4].permute(0, 2, 3, 1).reshape(-1, 65)
        assert bank.shape == (28928, 65)
        model.mpdd2_white_reference.copy_(bank)
        new(RUN / 'registration.json', {'UTC': utc(), 'normal_FIT_photos': 113, 'reference_vectors': len(bank),
            'actual_optimizer': 0, 'bank_SHA256': state_sha256({'bank': bank}), 'FIT_feature_cache_SHA256': sha(RUN / 'fit-features.pt')})

        def score(phase):
            rows, features = phase_features(phase)
            scores = []
            with torch.inference_mode():
                for index, value in enumerate(features):
                    scores.append(float(reference_evidence(value[None].cuda().float(), model.mpdd2_white_reference)[1][0]))
            return rows, scores

        _, val_scores = score('validation')
        new(RUN / 'normal-VAL-diagnostic.json', {'scores': val_scores, 'not_used_for_bank_or_selection': True})
        cal_rows, cal_scores = score('cal')
        cut, rank = normal_budget_threshold(cal_scores, alpha=.15)
        new(RUN / 'calibration.json', {'UTC': utc(), 'scores': cal_scores, 'normal_CAL': len(cal_rows),
                                      'threshold': cut, 'rank': rank, 'alpha': .15, 'saved_before_TEST': True})
        test_rows, test_scores = score('test')
        rates = classification_rates([row['label'] for row in test_rows], test_scores, cut)
        eligible = bool(rates['fpr'] <= .20 and rates['fnr'] <= .20)
        evaluation = {'UTC': utc(), **rates, 'AUROC': rank_auroc([row['label'] for row in test_rows], test_scores),
            'threshold': cut, 'eligible_research_candidate': eligible,
            'rows': [{'id': row['id'], 'group_id': row['group_id'], 'label': row['label'], 'score': score} for row, score in zip(test_rows, test_scores)],
            'normal_FIT_registration_photos': 113, 'reference_vectors': 28928, 'actual_optimizer': 0,
            'source_domain_only': True, 'no_pixel_GT': True, 'v40_public_TEST_already_seen': True,
            'independent_field_validation': False, 'app_adopted': False, 'new_API_CPP_ZIP_checks': 0, 'new_ZIPs': 0}
        new(RUN / 'evaluation.json', evaluation)
        new(ROOT / 'reports/industrial-mpdd2-local-reference-v41-evaluation.json', evaluation)
        current = model.state_dict()
        assert all(torch.equal(tensor, current[name].cpu()) for name, tensor in parent['state_dict'].items())
        candidate = deepcopy(parent)
        candidate['state_dict']['mpdd2_white_reference'] = bank.clone()
        candidate['mpdd2_white_reference_version'] = VERSION
        candidate['mpdd2_white_reference_protocol_SHA256'] = sha(PROTOCOL)
        delta = pack_delta(parent, candidate, prior.EXPECTED, prior.START.relative_to(ROOT).as_posix(), ('mpdd2_white_reference',))
        (RUN / 'registered').mkdir()
        checkpoint = RUN / 'registered/final-delta.pt'
        with checkpoint.open('xb') as output:
            torch.save(delta, output)
        restored = restore_delta(parent, torch.load(checkpoint, map_location='cpu', weights_only=True), prior.EXPECTED)
        assert candidate.keys() == restored.keys()
        assert all(torch.equal(tensor, restored['state_dict'][name]) for name, tensor in candidate['state_dict'].items())
        assert all(value == restored[name] for name, value in candidate.items() if name != 'state_dict')
        for name, expected in policy['legacy_Python_SHA256'].items():
            assert sha(ROOT / name) == expected, name
        for name, expected in read(RUN / 'protected-stat.json').items():
            stat = (ROOT / name).stat()
            assert [stat.st_size, stat.st_mtime_ns] == expected, name
        verify = {'UTC': utc(), 'parent_all_tensors_buffers_metadata_exact': True, 'serialized_delta_restore_exact': True,
            'checkpoint_SHA256': sha(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
            'parent_SHA256_unchanged': sha(prior.START) == prior.EXPECTED,
            'legacy_Python_SHA256_checked': len(policy['legacy_Python_SHA256']),
            'protected_files_stat_checked': len(read(RUN / 'protected-stat.json')), 'full_protected_SHA_recalculated': False,
            'fresh_encoder_RGB_inputs': 304, 'actual_optimizer': 0}
        new(RUN / 'verification.json', verify)
        new(ROOT / 'reports/industrial-mpdd2-local-reference-v41-verification.json', verify)
        print({key: value for key, value in evaluation.items() if key != 'rows'}, flush=True)
    except BaseException as error:
        new(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%H%M%S') + '.json'),
            {'UTC': utc(), 'error': repr(error), 'actual_optimizer': 0, 'partial_cache_preserved': True})
        raise
    finally:
        if read(lock)['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    globals()[parser.parse_args().stage]()
