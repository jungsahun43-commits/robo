"""Resolve labels/archive paths and author grouping before pixel/model work."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_gearbox_candidate import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    assert not (DATA/'metadata-labels-audit.json').exists()
    receipt=read(DATA/'download-receipt.json');intent=read(DATA/'split-intent-before-image-viewing.json')
    assert not intent['image_content_viewed']
    docs=DATA/'publisher-annotations';name='gbc_annotations_v20_functional_component_id.json'
    author=read(docs/name);fold=author['0'];train,test=fold['train'],fold['test']
    reference={r['idx']:r for r in train+test};assert len(reference)==15588
    assert {r['label_name'] for r in reference.values()}=={'OK','nOK'}
    assert all((r['label_int']==0 and r['label_name']=='OK') or (r['label_int']==1 and r['label_name']=='nOK') for r in reference.values())
    assert len({r['path'] for r in reference.values()})==15588
    consistency={}
    for path in sorted(docs.glob('*.json')):
        value=read(path)
        for key,f in value.items():
            rows=f['train']+f['test'];assert len(rows)==len(reference)
            assert len({r['idx'] for r in rows})==len(reference)
            assert all(r==reference[r['idx']] for r in rows)
            assert {r['idx'] for r in f['train']}.isdisjoint(r['idx'] for r in f['test'])
        consistency[path.name]={'folds':len(value),'all_records_labels_metadata_identical':True,'sha256':sha(path)}
    members={r['name']:r for r in receipt['members']};mapping={}
    for r in reference.values():
        # Interpret only the exact author path prefix as data; never join '../' onto disk.
        assert r['path'].startswith('../data/patched/')
        suffix=r['path'][len('../data/patched/'):];relative=PurePosixPath(suffix)
        assert not relative.is_absolute() and '..' not in relative.parts and not any(':' in p for p in relative.parts)
        archive_name='data/data/patched/'+relative.as_posix()
        assert archive_name in members and archive_name.endswith('.png')
        mapping[r['idx']]=archive_name
    assert len(set(mapping.values()))==15588
    unlabeled=sorted(n for n in members if n.endswith('.png') and n not in set(mapping.values()))
    assert len(unlabeled)==103
    ids=sorted(reference);parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    for field in ('physical_part_id','functional_component_id','acq_id'):
        seen={}
        for i in ids:
            key=reference[i][field]
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i in ids:groups[find(i)].append(i)
    testids={r['idx'] for r in test};bad={find(i) for i in testids}
    excluded=[r['idx'] for r in train if find(r['idx']) in bad]
    candidates=[r['idx'] for r in train if find(r['idx']) not in bad]
    intersections={field:sorted({r[field] for r in train}&{r[field] for r in test}) for field in ('physical_part_id','functional_component_id','acq_id')}
    result={'UTC':datetime.now(timezone.utc).isoformat(),'status':'metadata_checked_pixel_duplicates_and_final_split_pending',
        'publisher_ref':receipt['publisher_ref'],'publisher_version':1,'publisher_license':receipt['publisher_license'],
        'archive_sha256_local':receipt['archive_sha256_local'],'author_json_consistency':consistency,
        'annotated_unique_patches':15588,'annotated_label_counts':dict(Counter(r['label_name'] for r in reference.values())),
        'archive_PNGs':15691,'unannotated_PNGs_excluded':103,
        'publisher_physical_parts':len({r['physical_part_id'] for r in reference.values()}),
        'publisher_functional_component_ids':len({r['functional_component_id'] for r in reference.values()}),
        'publisher_acquisition_ids':len({r['acq_id'] for r in reference.values()}),
        'primary_holdout_reserved_before_image_viewing':'author functional_component_id fold0(S3)',
        'reserved_author_TEST_patches':len(test),'reserved_TEST_labels':dict(Counter(r['label_name'] for r in test)),
        'original_author_TRAIN_patches':len(train),'original_TRAIN_TEST_metadata_overlap':intersections,
        'connected_physical_functional_acquisition_groups':len(groups),
        'TRAIN_patches_linked_to_reserved_TEST_groups_must_be_excluded':len(excluded),
        'remaining_TRAIN_candidate_patches_before_pixel_audit':len(candidates),
        'remaining_TRAIN_candidate_groups_before_pixel_audit':len({find(i) for i in candidates}),
        'path_mapping':'exact ../data/patched/ prefix interpreted as data/data/patched/ archive member; no unsafe filesystem parent traversal',
        'source_normal_label_scope':'OK/nOK are source visible-wear acceptance labels; light scratches/pitting may still be OK, not every visible mark is a defect and not whole gearbox health',
        'mask_or_bbox_ground_truth_available':False,'new_independent_field_validation':False,
        'RGB_pixel_exact_orientation_near_audit_done':False,'current_retained_TEST_cross_duplicate_audit_done':False,
        'final_FIT_VAL_CAL_manifest_fixed':False,'model_features_scores_optimizer_updates':0,
        'image_content_viewed_by_agent':False,'author_or_archive_code_executed':False}
    new(DATA/'metadata-labels-audit.json',result)
    new(DATA/'reserved-author-S3-fold0.json',{'source_json_sha256':sha(docs/name),'TEST_ids':sorted(testids),
        'TRAIN_ids_linked_to_TEST_metadata_groups_excluded':sorted(excluded),'TRAIN_candidate_ids':sorted(candidates),
        'idx_to_archive_member':{str(k):v for k,v in mapping.items()},'unannotated_archive_members_excluded':unlabeled,
        'idx_to_metadata_group':{str(i):find(i) for i in ids}})
    new(ROOT/'reports/industrial-gearbox-v25-metadata-audit.json',result)
    print({k:v for k,v in result.items() if k!='author_json_consistency'},flush=True)


if __name__=='__main__':main()
