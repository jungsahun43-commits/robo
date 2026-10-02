"""Prepare original DACL photo tags from verified TRAIN full photos only."""
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha,dacl_items
from safelog_ai.auxiliary_classifier import AUX_CLASSES


def original_targets(doc):
    labels={s['label'] for s in doc['shapes']}
    if labels-set(AUX_CLASSES):raise ValueError('New source tag requires explicit vocabulary review')
    return [int(c in labels) for c in AUX_CLASSES]


def main():
    photos,_=dacl_items('train',640)
    source=ROOT/'data/dacl10k-yolo/records.json';records={r['stem']:r for r in read(source) if r['split']=='train'}
    if {Path(p['image']).stem for p in photos}!=set(records):raise ValueError('DACL TRAIN order/set mismatch')
    items=[];counts=Counter()
    for photo in photos:
        record=records[Path(photo['image']).stem];image=ROOT/'data'/record['source']
        annotation=image.parent.parent.parent/'annotations/train'/f'{image.stem}.json'
        targets=original_targets(read(annotation));counts.update(c for c,t in zip(AUX_CLASSES,targets) if t)
        items.append({'image':photo['image'],'targets':targets,'split':'train','domain':'dacl',
                      'annotation':annotation.relative_to(ROOT).as_posix(),'annotation_sha256':sha(annotation)})
    output=ROOT/'data/facility-auxiliary-training';output.mkdir(exist_ok=True)
    result={'split':'train','classes':list(AUX_CLASSES),'items':items,'audit':{'status':'prepared','full_train_photos':len(items),
            'positive_counts':dict(sorted(counts.items())),'records_sha256':sha(source),'script_sha256':sha(Path(__file__)),
            'policy':'Original TRAIN full-photo tags unchanged; non-DACL and detail crops have UNKNOWN auxiliary targets, never parent-level negatives'}}
    save(output/'train.json',result);save(ROOT/'reports/facility-target-auxiliary-data-audit.json',result['audit'])
    print(__import__('json').dumps(result['audit'],indent=2))


if __name__=='__main__':main()
