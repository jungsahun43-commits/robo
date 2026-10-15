"""v17 exclusion audit only. No inference, new labels, or source/cache mutation."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_magnetic_tiles import Bands
from safelog_ai.belt_normal_roi import fingerprints,distance
from scripts.fetch_equipment_expansion import sha
from scripts.study_equipment_expansion import read
from scripts.prepare_equipment_expansion import new


def matches(queries,candidates):
    index=Bands();exact=defaultdict(set)
    for i,r in enumerate(candidates):index.add(r['orientation_phashes'],i);exact[r['rgb_sha256']].add(i)
    found=[]
    for r in queries:
        ids=set(exact.get(r['rgb_sha256'],()))
        for i in index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],candidates[i]['orientation_phashes'])<=2:ids.add(i)
        found.extend({'query':r['id'],'query_phase':r['phase'],'candidate':candidates[i]['id'],'candidate_phase':candidates[i]['phase']} for i in sorted(ids))
    return found


def main():
    output=ROOT/'reports/industrial-mixed-real-mask-v17-data-audit.json';assert not output.exists()
    names=['data/equipment-expansion-v1/manifest.json','data/industrial-ksdd-supervised-v10/manifest.json','data/industrial-bsclass-v1/manifest.json','data/industrial-magnetic-tiles-v16/manifest.json']
    manifests=[read(ROOT/n) for n in names];old={}
    for manifest in manifests[:3]:
        for r in manifest['rows']:
            if r['phase'] in ('test','reserved_test'):old[r['image']]=dict(r)
    def prepare(r):
        assert sha(ROOT/r['image'])==r['image_sha256']
        if not r.get('orientation_phashes') or not r.get('rgb_sha256'):
            with Image.open(ROOT/r['image']) as im:
                rgb=im.convert('RGB');r['orientation_phashes']=fingerprints(rgb);r['rgb_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        return r
    with ThreadPoolExecutor(max_workers=4) as pool:old=list(pool.map(prepare,old.values()))
    ks=[r for r in manifests[1]['rows'] if r['phase'] in ('fit','validation')]
    mt=[r for r in manifests[3]['rows'] if r['phase'] in ('fit','validation','cal','test')]
    ks_to_old=matches(ks,old)
    cross=matches(ks,mt)
    bad_cross=[r for r in cross if r['query_phase']!=r['candidate_phase']]
    data={'UTC':datetime.now(timezone.utc).isoformat(),'input_manifests_sha256':{n:sha(ROOT/n) for n in names},
          'all_current_TEST_photos_checked':len(old),'KS_FIT_VAL_checked':len(ks),'MT_all_active_phases_checked':len(mt),
          'KS_FIT_VAL_matches_current_TEST':ks_to_old,'KS_FIT_VAL_matches_MT_all_phases':cross,'cross_phase_matches':bad_cross,
          'MT_vs_all_current_TEST_prior_audit':'industrial-magnetic-tiles-v16-data-audit.json (all1344 source photos,0matches)',
          'distance_rule':'exactRGB or any8orientation pHash<=2; conservative exclusion, not proof of physical identity',
          'ready':not ks_to_old and not bad_cross,'model_inference_performed':False,'optimizer_updates':0,
          'physical_acquisition_groups_verified':False,'independent_field_validation':False}
    new(output,data);print(data,flush=True)
    assert data['ready'],'Discovered exclusion links; redesign split/cache filtering before preregistration'


if __name__=='__main__':main()
