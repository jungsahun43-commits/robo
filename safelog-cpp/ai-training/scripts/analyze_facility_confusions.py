"""Source annotation co-occurrence in DACL validation errors, never gold edits."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha,TARGETS,dacl_items


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='facility-presence-target-spatial')
    args=parser.parse_args();run=ROOT/'runs'/args.name
    cache=read(run/'validation-dacl.json');val=read(run/'VALIDATION.json')
    items,classes=dacl_items('val',640)
    if cache['split']!='val' or not np.array_equal(cache['targets'],[i['targets'] for i in items]):
        raise ValueError('DACL validation item order/labels changed')
    if cache['weights_sha256']!=sha(run/'best.pt'):raise ValueError('Predictions not from frozen weights')
    records={r['stem']:r for r in read(ROOT/'data/dacl10k-yolo/records.json') if r['split']=='val'}
    source_labels=[];annotations=[]
    for i in items:
        r=records[Path(i['image']).stem];source=ROOT/'data'/r['source']
        path=source.parent.parent.parent/'annotations/train'/f'{source.stem}.json'
        doc=read(path);source_labels.append(set(s['label'] for s in doc['shapes']))
        annotations.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    t=np.asarray(cache['targets']);p=np.asarray(cache['probabilities']);output={}
    for label in TARGETS:
        k=classes.index(label);neg=t[:,k]==0;found=p[:,k]>=val['operating_points'][label]['threshold'];fp=neg&found
        counts=Counter(c for j in np.flatnonzero(fp) for c in source_labels[j]);cohorts=[]
        for original in sorted(set.union(*source_labels)):
            group=neg&np.array([original in labels for labels in source_labels])
            n=int(group.sum());bad=int((group&found).sum())
            if n:cohorts.append({'source_label':original,'negative_photos':n,'false_positives':bad,'fpr':bad/n})
        output[label]={'negative_photos':int(neg.sum()),'false_positives':int(fp.sum()),
                       'false_positive_source_labels':dict(counts.most_common()),'overlapping_cohorts':cohorts}
    result={'run':run.name,'split':'val','scope':'DACL original annotations; cohorts overlap; neither causal diagnosis nor corrected ground truth',
            'weights_sha256':cache['weights_sha256'],'predictions_sha256':sha(run/'validation-dacl.json'),
            'script_sha256':sha(Path(__file__)),'source_annotations':annotations,'per_class':output}
    save(ROOT/'reports'/f'{args.name}-confusion-audit.json',result)
    print(__import__('json').dumps({label:{k:v for k,v in data.items() if k!='overlapping_cohorts'} for label,data in output.items()},indent=2))


if __name__=='__main__':main()
