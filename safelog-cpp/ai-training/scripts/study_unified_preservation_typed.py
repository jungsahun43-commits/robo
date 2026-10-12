"""Correct dataset contract: known is a class vector, never an image coordinate."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_unified_preservation as study
from scripts.unified_main_data import Cache,read,sha

study.BASE=ROOT/'runs/unified-preservation-typed-v3'
study.PROTOCOL=ROOT/'reports/unified-preservation-typed-protocol.json'
study.LEDGER=study.BASE/'protected-inputs.json'
SOURCES=('scripts/study_unified_preservation_typed.py','tests/test_unified_preservation_typed.py')


class TypedCache(Cache):
    def batch(self,indices,device='cuda',training=False):
        indices=list(indices)
        images,truth=super().batch(indices,device,training)
        if 'pixel_known' in truth:
            original=torch.stack([self.row(int(i))[1]['pixel_known'] for i in indices]).to(device).float()
            if truth['pixel_known'].ndim!=2 or truth['pixel_known'].shape[1]!=7 or not torch.equal(truth['pixel_known'],original):
                raise ValueError('Spatial supervision known flags are Nx7 class flags; cannot flip or reorder classes')
        return images,truth


def freeze_typed():
    assert not study.BASE.exists();study.BASE.mkdir()
    original=ROOT/'runs/unified-preservation-v2/preflight.json'
    shutil.copy2(original,study.BASE/'preflight.json')
    study.freeze()
    study.new(ROOT/'reports/unified-preservation-known-scope.json',{
        'schema':'known-class-vector-contract-v3','declared_before_fit':True,
        'protocol_path':study.PROTOCOL.relative_to(ROOT).as_posix(),'protocol_sha256':sha(study.PROTOCOL),
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'actual_known_shape_per_item':[7],'actual_mask_shape_per_item':[7,80,80],
        'meaning':'known[c] enables spatial supervision for class c over the whole grid; not a spatial mask',
        'correct_augmentation':'Use preserved v1 Cache: flip image and pixel target mask, leave known class vector unchanged',
        'typed_cache_matches_previous_train_processing':True,'both_variants_use_same_correct_cache':True,
        'invalid_v2_partial_run_excluded':'unified-preservation-v2-aborted.json',
        'new_gold_or_unknown_to_negative_conversion':False,'tests_passed':2,
        'GPU_preflight_reused_from_v2':'Two disposable backbone freeze checks used only foreign task and remain valid; no new GPU preflight calls'})
    print('Typed class-vector contract frozen; old incorrect partial fit excluded',flush=True)


def activate():
    scope=read(ROOT/'reports/unified-preservation-known-scope.json')
    assert scope['protocol_sha256']==sha(study.PROTOCOL)
    repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name,digest in scope['source_sha256'].items():
        assert sha(ROOT/name)==digest
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==digest
    for path in [study.PROTOCOL,ROOT/'reports/unified-preservation-known-scope.json']:
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+path.relative_to(ROOT).as_posix()])).hexdigest()==sha(path)
    study.Cache=TypedCache;original_new=study.new
    def typed_new(path,value):
        if path.name.startswith('unified-preservation-') and path.name.endswith('-training.json'):
            value={**value,'known_class_vector_preserved':True,'known_scope_sha256':sha(ROOT/'reports/unified-preservation-known-scope.json'),
                   'protocol_path':study.PROTOCOL.relative_to(ROOT).as_posix(),'invalid_partial_fit_inherited':False}
        original_new(path,value)
    study.new=typed_new


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','train','validation','select','foreign','verify'])
    parser.add_argument('--variant',choices=study.VARIANTS);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='freeze':freeze_typed()
    else:
        activate()
        if args.mode=='train':
            for variant in ([args.variant] if args.variant else study.VARIANTS):study.train_variant(variant)
        else:{'validation':study.evaluate_validation,'select':study.select,'foreign':study.evaluate_foreign,'verify':study.verify}[args.mode]()
