"""Conservative local-feature overlap screen against existing held-out photos."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha,dacl_items,split_supplemental


def features(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError('Unreadable overlap-screen image')
    h,w=image.shape;scale=min(1,768/max(h,w))
    if scale<1:image=cv2.resize(image,(round(w*scale),round(h*scale)))
    orb=cv2.ORB_create(nfeatures=600,scaleFactor=1.25,nlevels=10)
    key,descriptor=orb.detectAndCompute(image,None)
    return np.array([p.pt for p in key],dtype=np.float32),descriptor


def geometric_overlap(left,right):
    lp,ld=left;rp,rd=right
    if ld is None or rd is None or min(len(ld),len(rd))<12:return 0
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ld,rd,k=2)
    good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
    if len(good)<12:return 0
    src=np.array([lp[m.queryIdx] for m in good]);dst=np.array([rp[m.trainIdx] for m in good])
    _,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3)
    count=int(inliers.sum()) if inliers is not None else 0
    if count<12 or count/len(good)<.5:return 0
    chosen=src[inliers.ravel().astype(bool)]
    if cv2.contourArea(cv2.convexHull(chosen))<100:return 0
    return count


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(48)
    source=ROOT/'data/s2ds-spatial-training/train.json';data=read(source)
    held=[]
    for split in ('val','test'):held+=dacl_items(split,640)[0]
    held+=split_supplemental()['val']+read(ROOT/'data/damsegment-training/test.json')['items']
    for split in ('val','test'):held+=read(ROOT/f'data/codebrim-training/{split}.json')['items']
    info=[];usable=[];records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(features,[ROOT/r['image'] for r in held])):
            if value[1] is not None and len(value[1])>=12:info.append(value);usable.append(held[i]['image'])
            if (i+1)%1000==0:print(f'S2DS held-out local features {i+1}/{len(held)}',flush=True)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=6,table_number=6,key_size=20,multi_probe_level=1),dict(checks=50))
    matcher.add([d for _,d in info]);matcher.train()
    exclusions=[];unverifiable=[]
    for i,item in enumerate(data['items']):
        query=features(ROOT/item['image'])
        if query[1] is None or len(query[1])<12:unverifiable.append(item['image']);continue
        votes=Counter(pair[0].imgIdx for pair in matcher.knnMatch(query[1],k=2) if pair)
        for j,_ in votes.most_common(20):
            inliers=geometric_overlap(query,info[j])
            if inliers:exclusions.append({'training_image':item['image'],'heldout_image':usable[j],'inliers':inliers,'group_id':item['group_id']});break
        if (i+1)%100==0:print(f'S2DS crop overlap screen {i+1}/{len(data["items"])}',flush=True)
    groups={e['group_id'] for e in exclusions}
    report={'source_manifest_sha256':sha(source),'heldout_photos':len(held),'heldout_with_features':len(usable),
            'local_overlaps':exclusions,'excluded_group_ids':sorted(groups),'low_feature_queries_unverifiable':unverifiable,
            'method':'ORB600, longest side768, 10 scales; FLANN-LSH6/20/1 top20 retrieval; Hamming ratio.7; RANSAC3px >=12 inliers, >=.5 match fraction, convex hull area>=100px',
            'limitation':'Heuristic overlap screen, not proof of scene/crop independence. Unverifiable low-feature queries are excluded before supplemental training.'}
    save(ROOT/'reports/facility-target-s2ds-crop-audit.json',report)
    kept=[i for i in data['items'] if i['group_id'] not in groups and i['image'] not in set(unverifiable)]
    counts=np.zeros((len(data['classes']),2),dtype=np.int64)
    for item in kept:
        with np.load(ROOT/item['pixel_target']) as mask:
            positive=(mask['mask']*mask['known'][:,None,None]).sum((1,2),dtype=np.int64)
            counts+=np.stack((positive,mask['known'].astype(np.int64)*6400-positive),axis=1)
    audit={**data['audit'],'usable_train_patches':len(kept),'local_feature_overlap_groups_excluded':len(groups),
           'low_feature_training_queries_excluded':len(unverifiable),'crop_audit_sha256':sha(ROOT/'reports/facility-target-s2ds-crop-audit.json'),
           'per_label_pixel_cells':{c:{'positive':int(counts[k,0]),'negative':int(counts[k,1])} for k,c in enumerate(data['classes'])},
           'per_label_train_counts':{c:{'positive':sum(i['targets'][k]==1 for i in kept),'negative':sum(i['targets'][k]==0 for i in kept),'unknown':sum(i['targets'][k]<0 for i in kept)} for k,c in enumerate(data['classes'])}}
    save(source.with_name('crop-screened-train.json'),{'split':'train','classes':data['classes'],'items':kept,'audit':audit})
    save(ROOT/'reports/facility-target-s2ds-screened-data-audit.json',audit)
    print('S2DS screened train patches',len(kept),'overlap groups',len(groups),'unverifiable',len(unverifiable),flush=True)


if __name__=='__main__':main()
