"""One real-mask/detail-view experiment with a fixed pre-training data gate."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.tile_surface import VERSION, tile_boxes, tile_view, per_tile_loss, DetailSurfaceMain
from safelog_ai.fine_target_surface import trainable_fine_head
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.magnetic_surface import surface_view
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.belt_normal_roi import fingerprints, distance
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_unified_belt_tiles import snapshot
from scripts.prepare_magnetic_tiles import Bands
from scripts.study_mixed_real_mask import get_cache as old_cache, parent_target_scores, draw_group_balanced
from scripts.study_real_mask_transfer import group_rates

RUN = ROOT / 'runs/industrial-neu-detail-v22'
DATA = ROOT / 'data/industrial-neu-detail-v22'
NEU = ROOT / 'data/industrial-neu-micro-v22'
REPORTS = ROOT / 'reports'
PROTOCOL = REPORTS / 'industrial-neu-detail-v22-protocol.json'
MANIFEST = DATA / 'manifest.json'
NORMAL = ROOT / 'data/industrial-neighbor-surfaces-v15/manifest.json'
START = ROOT / 'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED = '5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
WARM = ROOT / 'runs/industrial-balanced-targets-v21/trained/final.pt'
WARM_SHA = '43df7c83403d65cbe0669e59c30c9c1085c0936de2e6c8b611ace436923dabdc'
PARENT_EVAL = START.parent.parent / 'evaluation.json'
TARGETS = ('mpdd_bracket_black', 'mpdd_tubes')
METHODS = ('detail_only', 'max_detail_parent')
SOURCES = ('safelog_ai/tile_surface.py', 'scripts/study_neu_detail.py', 'tests/test_tile_surface.py')
EPOCHS, STEPS, BATCH, SEED = 30, 64, 8, 2222
KEYS = ('low', 'high', 'fine', 'masks', 'valid')
HELDOUT = ('data/equipment-expansion-v1/manifest.json', 'data/industrial-ksdd-supervised-v10/manifest.json',
           'data/industrial-bsclass-v1/manifest.json', 'data/industrial-magnetic-tiles-v16/manifest.json',
           'data/industrial-ksdd1-v18/manifest-r1.json')


def freeze():
    assert not RUN.exists() and not DATA.exists() and not PROTOCOL.exists()
    assert sha(START) == EXPECTED and sha(WARM) == WARM_SHA
    assert read(ROOT / 'runs/industrial-balanced-targets-v21/selection.json')['selected'] == {}
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3
    RUN.mkdir(); DATA.mkdir()
    ledger = dict(read(ROOT / 'runs/industrial-balanced-targets-v21/protected-inputs.json'))
    for folder in (ROOT / 'runs/industrial-balanced-targets-v21', NEU, REPORTS):
        for path in folder.rglob('*'):
            name = path.relative_to(ROOT).as_posix()
            if path.is_file() and not path.name.endswith(('.lock', '.log')) and name not in ledger:
                ledger[name] = snapshot(path)
    new(RUN / 'protected-inputs.json', ledger)
    p = {'schema': VERSION, 'UTC': datetime.now(timezone.utc).isoformat(),
        'fixed_before_new_data_tiles_features_gradients_CAL_TEST_scores': True,
        'source_sha256': {n: sha(ROOT / n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('safelog_ai', 'scripts', 'tests') for p in (ROOT / d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'parent_sha256': EXPECTED, 'warm_start_sha256': WARM_SHA, 'parent_evaluation_sha256': sha(PARENT_EVAL),
        'normal_manifest_sha256': sha(NORMAL), 'NEU_candidate_manifest_sha256': sha(NEU / 'candidate-manifest.json'),
        'NEU_FIT_review_sha256': sha(NEU / 'FIT-review-r1.json'), 'NEU_QA_reservation_sha256': sha(NEU / 'QA-reservation.json'),
        'protected_inputs_sha256': sha(RUN / 'protected-inputs.json'), 'heldout_manifests_sha256': {n: sha(ROOT / n) for n in HELDOUT},
        'source_split': 'NEU_SDI20+SPDI15 natural-mask photos FIT; OilSmall4+Wipe6 VAL; already-viewed OilScratch6 excluded, never moved into VAL/TEST',
        'normal_split': 'inherited MPDD publisher normal FIT black70/tubes61, VAL18/15; CAL30/23 unchanged, TEST79/101 unchanged',
        'new_source_normal_whole_photos': 0, 'new_app_category_claimed': False,
        'tile_policy': 'all nine half-width/half-height native RGB crops, positions start/middle/end on each axis, ceil half extent; identical at train/VAL/CAL/TEST; each crop aspect-preserving512 RGB bicubic letterbox127 and nearest original mask',
        'source_patch_labels': 'native source mask crop empty = annotation-negative crop only; >=32 positive pixels at512 = positive; 1..31 excluded, never relabelled normal. Publisher normal photos yield normal crops',
        'near_audit': 'raw crop and512view RGB SHA and union of 8orientation pHashes; exactRGB or min pHash<=2. FIT photo/appearance groups union via all FIT crop matches, exclude entire connected original-group if any crop matches VAL/CAL/TEST raw full or declared crops. No TEST masks/scores used',
        'data_gate': 'FIT>=20 real-mask original photos, >=5 each SDI/SPDI with usable positives; annotation-negative crops each source>0; target normal original FIT>=30 each; VAL>=8 real-mask positive photos and target normal>=10 each; all4VALbuckets nonempty; else stop before features/optimizer',
        'features': 'one frozen original encoder, fine24@128 low40@64 high960@32, FP16 roundtrip; no new encoder; same512tile coordinates/mask/valid checked',
        'head': 'v21 FineTargetHead89channels at128grid copied to distinct detail_surface_head; zero residual channel for EVERY source/category/inference. No equipment category or source embedding. Raw fine no new normalization',
        'trainable': 'new fine projection and fusion only; high projection and every original parent tensor/buffer/metadata frozen',
        'loss': 'per-tile positive/negative class-mean BCE(.5/.5 for positive, negative mean for annotation-negative) +softDice+.5top32valid photoBCE. No area positive-weight multiplier',
        'sampling': 'batch8:2SDI positive,2SPDI positive,1SDI annotation-negative,1SPDI annotation-negative,1black normal,1tubes normal; uniform original photo then uniform eligible tile via inverse tile multiplicity; replacement not new photos',
        'augmentation': 'paired cached fine/low/high/mask/valid H/V flips .5, not fresh flipped RGB encoder passes',
        'optimizer': 'AdamW lr1e-4 wd1e-4 cosine30 min1e-6 clip5 FP32 TF32off',
        'epochs': EPOCHS, 'steps': STEPS, 'batch': BATCH, 'seed': SEED, 'planned_optimizer_updates': EPOCHS * STEPS,
        'VAL_selection': 'initial epoch0 and each completed epoch, macro four buckets realpositive/annotationnegative/blacknormal/tubenormal; mean loss per originalphoto then mean photos per bucket; earliest strict minimum, family-heldout source not independent field',
        'resume': 'per-epoch optimizer/scheduler/RNG, append actual step journal and count replay; completed studies never rerun',
        'photo_score': 'maximum of all nine valid top32 sigmoid crop scores; count original photo ONCE, never treat crops as independent evaluation photos',
        'CAL': 'normal photos30/23;85percent order ceil((n+1)*.85),strict >; detail_only and max(new/newCALcut,currentv16/currentcut), own joint normalCALcut saved before TEST model scores',
        'methods': list(METHODS), 'guards': ['each TEST label>=20','FPR<=.20','FPR<=v16+.05','FNR<=v16-.05','subtype n>=10 FNR<=v16+.05','selected epoch>0'],
        'selection': 'eligible min maxerror then FNR then method order; no threshold retuning/guard relaxation',
        'location': 'original target primary PNG/pixelthresholds unchanged; new tile maps not promoted as default locations',
        'known_public_TEST_reused_for_adoption': True, 'independent_field_validated': False,
        'physical_capture_groups_verified': False, 'inherited_parent_past_near_exposure_undone': False,
        'new_source_masks_team_exhaustively_verified': False, 'new_unit_tests_passed': 4,
        'source_use': 'NEU original images plus abin24 masks, academic research only/no commercial/cite original provider and curator; no invented SPDX; preserve raw/failed artifacts',
        'release_gate': 'fresh packaged CPU/GPU/all aliases/C++ and original location PNG checks only if selected'}
    new(PROTOCOL, p); print({'frozen': True, 'planned_optimizer_updates': EPOCHS * STEPS}, flush=True)


def validate(committed=False, data=False):
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT / n) == h, n
    for path, h in ((START, EXPECTED), (WARM, WARM_SHA), (NORMAL, p['normal_manifest_sha256']),
        (PARENT_EVAL, p['parent_evaluation_sha256']), (NEU/'candidate-manifest.json', p['NEU_candidate_manifest_sha256']),
        (NEU/'FIT-review-r1.json', p['NEU_FIT_review_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, path
    if data:
        assert sha(MANIFEST) == read(RUN/'data-ready.json')['manifest_sha256']
        assert read(MANIFEST)['audit']['ready']
    if committed:
        repo = ROOT.parents[1]
        git = ['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
        names = [*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()]
        if data: names.append('reports/industrial-neu-detail-v22-data-audit.json')
        for n in names:
            assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n), n
    return p


def photo_rows():
    review = read(NEU/'FIT-review-r1.json'); rows = []
    fit, val = set(review['eligible_FIT_candidate_ids']), set(review['unviewed_VAL_candidate_ids'])
    for r in read(NEU/'candidate-manifest.json')['rows']:
        if r['id'] in fit | val:
            rows.append({**r,'phase':'fit' if r['id'] in fit else 'validation','is_NEU':True,'group_id':r['id']})
    for r in read(NORMAL)['rows']:
        if r['category'] in TARGETS and r['phase'] in ('fit','validation','cal','test'):
            if r['phase'] in ('fit','validation','cal'): assert r['label'] == 0
            rows.append({**r,'is_NEU':False,'group_id':r['appearance_group_proxy']})
    return rows


def fingerprint(rgb):
    view = surface_view(rgb)[0]
    return {'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'view_rgb_sha256': hashlib.sha256(view.tobytes()).hexdigest(),
            'orientation_phashes': sorted(set(fingerprints(rgb) + fingerprints(view)))}


def prepare_photo(row):
    assert sha(ROOT/row['image']) == row['image_sha256']
    with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
    truth = None
    if row['is_NEU']:
        assert sha(ROOT/row['mask']) == row['mask_sha256']
        with Image.open(ROOT/row['mask']) as im: truth = im.convert('L')
    patches = []
    for i, box in enumerate(tile_boxes(rgb.size)):
        view, mask, valid, geom = tile_view(rgb, box, truth)
        count = int(mask.sum())
        active = not row['is_NEU'] or count == 0 or count >= 32
        bucket = ('real_positive' if count else 'annotation_negative') if row['is_NEU'] else row['category']
        patches.append({'id':row['id']+f'/tile{i}','photo_id':row['id'],'group_id':row['group_id'],
            'image':row['image'],'image_sha256':row['image_sha256'],'is_NEU':row['is_NEU'],
            'mask':row.get('mask') if row['is_NEU'] else None,'mask_sha256':row.get('mask_sha256') if row['is_NEU'] else None,
            'source_folder':row.get('source_folder'),'phase':row['phase'],'category':row.get('category','NEU_surface_source'),
            'box':list(box),'geometry':geom,'positive_pixels':None if row['phase']=='test' else count,
            'label':None if row['phase']=='test' else int(count>0),'bucket':bucket,
            'active_support':active,'support_scope':'publisher annotated crop, not whole equipment health', **fingerprint(rgb.crop(box))})
    return patches


def prepare():
    p = validate(committed=True)
    assert not MANIFEST.exists() and not (RUN/'data-ready.json').exists()
    photos = photo_rows()
    with ThreadPoolExecutor(max_workers=4) as pool:
        crops = [r for group in pool.map(prepare_photo, photos) for r in group]
    print({'crop_coordinates_prepared':len(crops),'original_photos':len(photos)}, flush=True)
    heldout = {}
    for name, h in p['heldout_manifests_sha256'].items():
        assert sha(ROOT/name) == h
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'): heldout[r['image']] = dict(r)
    for r in photos:
        if r['phase'] != 'fit': heldout[r['image']] = dict(r)
    def full(row):
        assert sha(ROOT/row['image']) == row['image_sha256']
        with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
        return {'id':row['id'],'phase':row['phase'], **fingerprint(rgb)}
    with ThreadPoolExecutor(max_workers=4) as pool: refs = list(pool.map(full, heldout.values()))
    retained_TEST_full = sum(r['phase'] in ('test','reserved_test') for r in refs)
    refs.extend(r for r in crops if r['phase'] != 'fit')
    index = Bands(); exact = defaultdict(set)
    for i, r in enumerate(refs):
        index.add(r['orientation_phashes'], i)
        for field in ('rgb_sha256','view_rgb_sha256'): exact[r[field]].add(i)
    fit = [r for r in crops if r['phase']=='fit']; groups = sorted({r['group_id'] for r in fit})
    parent = {g:g for g in groups}
    def find(g):
        while parent[g]!=g: parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    local = Bands(); local_exact = defaultdict(set); direct=[]; internal=[]
    for i, r in enumerate(fit):
        candidates=set()
        for field in ('rgb_sha256','view_rgb_sha256'): candidates.update(local_exact[r[field]])
        for j in local.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],fit[j]['orientation_phashes'])<=2: candidates.add(j)
        for j in sorted(candidates):
            if r['group_id']!=fit[j]['group_id']:
                union(r['group_id'],fit[j]['group_id']);internal.append({'a':r['id'],'b':fit[j]['id']})
        local.add(r['orientation_phashes'],i)
        for field in ('rgb_sha256','view_rgb_sha256'): local_exact[r[field]].add(i)
        match=set()
        for field in ('rgb_sha256','view_rgb_sha256'): match.update(exact[r[field]])
        for j in index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],refs[j]['orientation_phashes'])<=2: match.add(j)
        direct.extend({'fit_crop':r['id'],'group_id':r['group_id'],'heldout_id':refs[j]['id'],'heldout_phase':refs[j]['phase']} for j in sorted(match))
    bad_roots={find(r['group_id']) for r in direct}; bad_groups={g for g in groups if find(g) in bad_roots}
    for r in crops:
        r['training_active'] = r['phase']=='fit' and r['group_id'] not in bad_groups and r['active_support']
        r['validation_active'] = r['phase']=='validation' and r['active_support']
    activefit=[r for r in crops if r['training_active']]; activeval=[r for r in crops if r['validation_active']]
    realphotos={r['photo_id'] for r in activefit if r['is_NEU'] and r['label']}
    counts={'fit':dict(Counter(r['bucket'] for r in activefit)),'validation':dict(Counter(r['bucket'] for r in activeval))}
    real_by_source={c:len({r['photo_id'] for r in activefit if r['is_NEU'] and r['label'] and r['source_folder']==c}) for c in ('NEU_SDI','NEU_SPDI')}
    normalcounts={phase:{c:len({r['photo_id'] for r in values if r['bucket']==c}) for c in TARGETS} for phase,values in [('fit',activefit),('validation',activeval)]}
    valreal=len({r['photo_id'] for r in activeval if r['is_NEU'] and r['label']})
    ready=(len(realphotos)>=20 and min(real_by_source.values())>=5 and all(any(r['source_folder']==c and not r['label'] for r in activefit if r['is_NEU']) for c in real_by_source)
        and min(normalcounts['fit'].values())>=30 and min(normalcounts['validation'].values())>=10 and valreal>=8
        and all(counts['validation'].get(b,0)>0 for b in ('real_positive','annotation_negative',*TARGETS)))
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'ready':ready,'protocol_sha256':sha(PROTOCOL),
        'photos_before_crop_exclusion':len(photos),'crop_coordinates':len(crops),'training_crops':len(activefit),'validation_crops':len(activeval),
        'counts':counts,'FIT_real_mask_original_photos':len(realphotos),'FIT_real_by_source':real_by_source,'VAL_real_mask_original_photos':valreal,
        'normal_original_photo_counts':normalcounts,'excluded_original_groups':sorted(bad_groups),'direct_heldout_near_links':direct,
        'FIT_internal_near_links':internal,'retained_TEST_full_images_checked':retained_TEST_full,
        'reference_full_and_crops_checked':len(refs),'excluded_small_positive_crops':sum(not r['active_support'] for r in crops if r['phase'] in ('fit','validation')),
        'TEST_masks_used_for_preparation':False,'new_model_scores_or_optimizer_updates':0,'physical_capture_groups_verified':False,
        'whole_original_groups_excluded_transitively':True,'no_new_source_normal_whole_photo_FPR':True,
        'new_unit_tests_passed':4,'source_phase_reservations_and_marker_exclusion_preserved':True}
    new(MANIFEST,{'schema':VERSION,'photos':photos,'rows':crops,'audit':audit})
    new(REPORTS/'industrial-neu-detail-v22-data-audit.json',audit)
    new(RUN/'data-ready.json',{'ready':ready,'manifest_sha256':sha(MANIFEST),'audit':audit})
    print({k:v for k,v in audit.items() if k not in ('direct_heldout_near_links','FIT_internal_near_links')},flush=True)
    assert ready,'Data guard failed; preserve preparation, do not run gradients or tune based on TEST'


def load(path=START,device='cpu',initialize=False):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=DetailSurfaceMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and (not missing or initialize and all(n.startswith('detail_surface_head.') for n in missing))
    if initialize:
        warm=torch.load(WARM,map_location='cpu',weights_only=True);prefix='fine_target_head.'
        model.detail_surface_head.load_state_dict({n[len(prefix):]:v for n,v in warm['state_dict'].items() if n.startswith(prefix)},strict=True)
    return model.eval().requires_grad_(False).to(device),state


def image_batch(rows):
    images=[];masks=[];valids=[]
    for r in rows:
        assert sha(ROOT/r['image'])==r['image_sha256']
        with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
        truth=None
        if r['is_NEU']:
            assert sha(ROOT/r['mask'])==r['mask_sha256']
            with Image.open(ROOT/r['mask']) as im:truth=im.convert('L')
        view,mask,valid,geom=tile_view(rgb,tuple(r['box']),truth)
        assert geom==r['geometry'] and hashlib.sha256(view.tobytes()).hexdigest()==r['view_rgb_sha256']
        assert int(mask.sum())==r['positive_pixels'] and not bool(mask[~valid].any())
        images.append(normalized(view));masks.append(mask);valids.append(valid)
    return torch.stack(images),torch.stack(masks),torch.stack(valids)


def cache(phase,model):
    active='training_active' if phase=='fit' else 'validation_active'
    rows=[r for r in read(MANIFEST)['rows'] if r[active]];path=RUN/(phase+'-features.pt')
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache.json'))['sha256']
        values=torch.load(path,map_location='cpu',weights_only=True)
        assert values['ids']==[r['id'] for r in rows] and values['manifest_sha256']==sha(MANIFEST)
        return rows,values
    values={k:[] for k in KEYS}
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            images,masks,valid=image_batch(rows[start:start+BATCH])
            f=forward_with_fine(model.backbone,images.cuda())
            for k in ('low','high','fine'):values[k].append(f[k].half().cpu())
            values['masks'].append(masks);values['valid'].append(valid)
            if start%256==0:print({'cache':phase,'crops':min(start+BATCH,len(rows)),'total':len(rows)},flush=True)
    saved={k:torch.cat(v) for k,v in values.items()};saved.update(ids=[r['id'] for r in rows],manifest_sha256=sha(MANIFEST),protocol_sha256=sha(PROTOCOL))
    partial=path.with_suffix('.partial.pt')
    with partial.open('xb') as out:torch.save(saved,out)
    partial.rename(path);new(RUN/(phase+'-cache.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'crops':len(rows),'manifest_sha256':sha(MANIFEST)})
    return rows,saved


def logits(head,b):
    zero=b['low'].new_zeros((len(b['low']),1,*b['high'].shape[-2:]))
    return torch.nn.functional.interpolate(head(b['low'],b['high'],zero,b['fine']),size=(512,512),mode='bilinear',align_corners=False)


def validation(head,rows,values):
    groups=defaultdict(lambda:defaultdict(list));head.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            at=slice(start,start+BATCH);b={k:values[k][at].cuda() for k in KEYS}
            losses=per_tile_loss(logits(head,b),b['masks'][:,None],b['valid'][:,None]).cpu().tolist()
            for r,loss in zip(rows[at],losses):groups[r['bucket']][r['photo_id']].append(loss)
    means={g:float(np.mean([np.mean(v) for v in photos.values()])) for g,photos in groups.items()}
    assert set(means)=={'real_positive','annotation_negative',*TARGETS}
    return {'group_photo_mean_losses':means,'balanced_loss':float(np.mean(list(means.values()))),'source':'actual-mask family-heldout VAL plus original target normal VAL'}


def train():
    p=validate(committed=True,data=True)
    assert not (RUN/'trained/training.json').exists() and shutil.disk_usage(ROOT).free>8*1024**3
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model,parent=load(device='cuda',initialize=True);head=model.detail_surface_head
    fr,fit=cache('fit',model);vr,val=cache('validation',model);params=trainable_fine_head(head)
    assert all(n.startswith('detail_surface_head.') for n,t in model.named_parameters() if t.requires_grad)
    draws=RUN/'draws.npz'
    if not draws.exists():
        rng=np.random.default_rng(SEED);pools=[]
        for folder,label,count in [('NEU_SDI',1,2),('NEU_SPDI',1,2),('NEU_SDI',0,1),('NEU_SPDI',0,1)]:
            ids=[i for i,r in enumerate(fr) if r['is_NEU'] and r['source_folder']==folder and r['label']==label]
            pools.extend([ids]*count)
        pools.extend([[i for i,r in enumerate(fr) if r['bucket']==c] for c in TARGETS]);assert len(pools)==8 and all(pools)
        sample_rows=[{**r,'group_id':r['photo_id']} for r in fr]
        indices=[[draw_group_balanced(sample_rows,ids,rng) for ids in pools] for _ in range(EPOCHS*STEPS)]
        np.savez_compressed(draws,indices=np.array(indices).reshape(EPOCHS,STEPS,BATCH),flips=rng.random((EPOCHS,STEPS,BATCH,2))<.5)
        new(RUN/'draws.json',{'sha256':sha(draws),'manifest_sha256':sha(MANIFEST)})
    assert sha(draws)==read(RUN/'draws.json')['sha256'];draws=np.load(draws)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    folder=RUN/'train';folder.mkdir(exist_ok=True);progress=folder/'progress.pt';journal=folder/'optimizer.jsonl';epoch0=0;actual=0;best_epoch=0
    if progress.exists():
        s=torch.load(progress,map_location='cpu',weights_only=True);assert s['protocol_sha256']==sha(PROTOCOL)
        head.load_state_dict(s['head']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);epoch0=s['epoch'];best=s['best'];best_epoch=s['best_epoch'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state(s['cuda_rng'])
    else:
        initial=validation(head,vr,val);best=initial['balanced_loss'];new(folder/'initial-VAL.json',initial)
        torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':0,'VAL_loss':best},folder/'best.pt')
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':0,'best':best,'best_epoch':0,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        print({'initial_VAL':initial},flush=True)
    if journal.exists():actual=sum(1 for _ in journal.open(encoding='utf-8'))
    for epoch in range(epoch0,EPOCHS):
        head.train();losses=[]
        for step in range(STEPS):
            at=draws['indices'][epoch,step];b={k:fit[k][at].clone().cuda() for k in KEYS}
            for i in range(BATCH):
                axes=tuple(axis for enabled,axis in zip(draws['flips'][epoch,step,i],(-1,-2)) if enabled)
                if axes:
                    for k in b:b[k][i]=b[k][i].flip(axes)
            optimizer.zero_grad(set_to_none=True);loss=per_tile_loss(logits(head,b),b['masks'][:,None],b['valid'][:,None]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step();actual+=1;losses.append(float(loss.detach()))
            with journal.open('a',encoding='utf-8') as out:out.write(json.dumps({'actual_operation':actual,'epoch':epoch+1,'step':step+1,'loss':losses[-1]})+'\n')
        vl=validation(head,vr,val)
        if vl['balanced_loss']<best:
            best=vl['balanced_loss'];best_epoch=epoch+1;torch.save({'head':{n:t.detach().cpu() for n,t in head.state_dict().items()},'epoch':best_epoch,'VAL_loss':best},folder/'best.pt')
        scheduler.step();torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'protocol_sha256':sha(PROTOCOL)},progress)
        item={'epoch':epoch+1,'actual_optimizer_updates':actual,'training_loss':float(np.mean(losses)),'VAL':vl,'selected_epoch':best_epoch}
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(item)+'\n')
        print(item,flush=True)
    chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);state=deepcopy(parent)
    state['state_dict'].update({'detail_surface_head.'+n:t for n,t in chosen['head'].items()})
    state.update(detail_surface_version=VERSION,detail_surface_parent_sha256=EXPECTED,detail_surface_warm_sha256=WARM_SHA,detail_surface_protocol_sha256=sha(PROTOCOL),detail_surface_actual_optimizer_updates=actual,detail_surface_selected_epoch=best_epoch)
    (RUN/'trained').mkdir();torch.save(state,RUN/'trained/final.pt')
    new(RUN/'trained/training.json',{'actual_optimizer_updates':actual,'scheduled_unique_updates':EPOCHS*STEPS,'completed_epochs':EPOCHS,'selected_epoch':best_epoch,'selected_epoch_updates':best_epoch*STEPS,'VAL_balanced_loss':best,'FIT_crops':len(fr),'VAL_crops':len(vr),'FIT_real_original_photos':read(MANIFEST)['audit']['FIT_real_mask_original_photos'],'manifest_sha256':sha(MANIFEST)})


def photo_scores(model,rows,device='cuda'):
    result=[]
    with torch.inference_mode():
        for row in rows:
            assert sha(ROOT/row['image'])==row['image_sha256']
            with Image.open(ROOT/row['image']) as im:rgb=im.convert('RGB')
            images=[];valid=[]
            for box in tile_boxes(rgb.size):
                view,_,known,_=tile_view(rgb,box);images.append(normalized(view));valid.append(known)
            crop_scores=[]
            for start in range(0,len(images),BATCH):
                prob=model.forward_detail_view(torch.stack(images[start:start+BATCH]).to(device)).sigmoid().cpu()
                crop_scores.extend(ksdd_photo_score(prob,torch.stack(valid[start:start+BATCH])).tolist())
            assert len(crop_scores)==9
            result.append({'id':row['id'],'score':max(crop_scores),'crop_scores':crop_scores})
    return result


def calibrate():
    validate(data=True);assert not (RUN/'calibration.json').exists();model,state=load(RUN/'trained/final.pt','cuda');result={}
    for c in TARGETS:
        rows,values=old_cache(c,'cal');assert not any(r['label'] for r in rows)
        observations=photo_scores(model,rows);scores=[r['score'] for r in observations];cut,rank=normal_budget_threshold(scores,.15)
        old,oldcut=parent_target_scores(model,state,c,values);assert min(cut,oldcut)>0
        joint,jrank=normal_budget_threshold([max(n/cut,o/oldcut) for n,o in zip(scores,old)],.15)
        result[c]={'points':{'detail_only':{'photo_threshold':cut,'rank':rank},'max_detail_parent':{'photo_threshold':joint,'rank':jrank,'detail_divisor':cut,'parent_divisor':oldcut}},'normal_CAL_photos':len(rows),'observations':[{**r,'parent_score':o} for r,o in zip(observations,old)]}
    new(RUN/'calibration.json',{'protocol_sha256':sha(PROTOCOL),'thresholds_saved_before_TEST':True,'categories':result})
    print({'calibrated':True,'points':{c:result[c]['points'] for c in TARGETS}},flush=True)


def evaluate():
    validate(data=True);assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json')
    assert cal['thresholds_saved_before_TEST'] and cal['protocol_sha256']==sha(PROTOCOL)
    model,state=load(RUN/'trained/final.pt','cuda');prior={r['id']:r for r in read(PARENT_EVAL)['rows']};result={};selected={};out=[]
    trained=read(RUN/'trained/training.json')['selected_epoch']>0
    for c in TARGETS:
        rows,values=old_cache(c,'test');labels=[r['label'] for r in rows];observations=photo_scores(model,rows);scores=[r['score'] for r in observations]
        old,oldcut=parent_target_scores(model,state,c,values);oldpred=[s>oldcut for s in old]
        assert oldpred==[prior[r['id']]['candidate_detected'] for r in rows]
        base=classification_rates(labels,oldpred,.5);bg=group_rates(rows,oldpred);result[c]={'parent':base,'parent_by_defect':bg,'methods':{}}
        for method in METHODS:
            point=cal['categories'][c]['points'][method]
            score=scores if method=='detail_only' else [max(n/point['detail_divisor'],o/point['parent_divisor']) for n,o in zip(scores,old)]
            pred=[s>point['photo_threshold'] for s in score];r=classification_rates(labels,pred,.5);groups=group_rates(rows,pred)
            eligible=trained and min(r['positive_photos'],r['negative_photos'])>=20 and r['fpr']<=.20 and r['fpr']<=base['fpr']+.05 and r['fnr']<=base['fnr']-.05 and all(g['photos']<10 or g['fnr']<=bg[d]['fnr']+.05 for d,g in groups.items())
            result[c]['methods'][method]={'rates':r,'by_defect':groups,'eligible':eligible,'auroc':rank_auroc(labels,score)}
            out.extend({'id':row['id'],'category':c,'method':method,'label':row['label'],'defect':row['defect'],'score':s,'candidate_detected':q,'detail_crop_scores':obs['crop_scores']} for row,s,q,obs in zip(rows,score,pred,observations))
        eligible=[m for m in METHODS if result[c]['methods'][m]['eligible']]
        if eligible:selected[c]=min(eligible,key=lambda m:(max(result[c]['methods'][m]['rates']['fpr'],result[c]['methods'][m]['rates']['fnr']),result[c]['methods'][m]['rates']['fnr'],METHODS.index(m)))
    s={'selected':selected,'categories':result,'training':read(RUN/'trained/training.json'),'CAL_sha256':sha(RUN/'calibration.json'),'known_public_TEST_reused':True,'independent_field_validated':False,'original_photos_not_crops_counted':True,'primary_location_unchanged':True}
    new(RUN/'candidate-evaluation.json',{'rows':out,'categories':result});new(RUN/'selection.json',s);new(REPORTS/'industrial-neu-detail-v22-selection.json',s)
    if selected:
        state['detail_surface_methods']=selected;state['detail_surface_points']={c:cal['categories'][c]['points'][m] for c,m in selected.items()};(RUN/'selected').mkdir();torch.save(state,RUN/'selected/final.pt')
    else:new(RUN/'retained-parent.json',{'checkpoint':str(START),'sha256':EXPECTED,'reason':'No preregistered detail candidate eligible'})
    print({'selection':s},flush=True)


def verify():
    p=validate(data=True);s=read(RUN/'selection.json');t=read(RUN/'trained/training.json')
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    unchanged=0;rehashed=0
    for n,before in read(RUN/'protected-inputs.json').items():
        path=ROOT/n;st=path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']:unchanged+=1
        else:assert sha(path)==before['sha256'],n;rehashed+=1
    parent=torch.load(START,map_location='cpu',weights_only=True);candidate=torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True);warm=torch.load(WARM,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,candidate['state_dict'][n]) for n,v in parent['state_dict'].items())
    assert all(candidate[k]==v for k,v in parent.items() if k!='state_dict')
    for n,v in warm['state_dict'].items():
        if n.startswith('fine_target_head.high_projection.'):assert torch.equal(v,candidate['state_dict'][n.replace('fine_target_head.','detail_surface_head.',1)])
    journal=[json.loads(line) for line in (RUN/'train/optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_operation'] for r in journal]==list(range(1,len(journal)+1)) and len(journal)==t['actual_optimizer_updates']
    for phase in ('fit','validation'):assert sha(RUN/(phase+'-features.pt'))==read(RUN/(phase+'-cache.json'))['sha256']
    result={'status':'passed','parent_all_tensors_buffers_metadata_exact':True,'high_projection_exact_to_v21':True,
        'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,
        'actual_optimizer_updates':len(journal),'new_unit_tests_passed':4,'selected':s['selected'],'original_primary_locations_unchanged':True,
        'CPU_GPU_CPP_release_checks_still_required_if_selected':bool(s['selected']),'active_v16_retained':True,'independent_field_validated':False}
    new(RUN/'verification.json',result);new(REPORTS/'industrial-neu-detail-v22-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','prepare','train','calibrate','evaluate','verify','run'));args=parser.parse_args()
    if args.mode in ('prepare','train','calibrate','evaluate','run'):
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(path))
        lock=RUN/'operation.lock'
        if lock.exists():new(RUN/('stale-lock-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'pid':lock.read_text(encoding='utf-8'),'live':False});lock.unlink()
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if args.mode=='run':train();calibrate();evaluate();verify()
            else:globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid(),'mode':args.mode});raise
        finally:lock.unlink()
    else:globals()[args.mode]()
