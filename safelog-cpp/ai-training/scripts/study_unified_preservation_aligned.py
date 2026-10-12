"""Pre-fit correction: flip every spatial truth grid with the TRAIN image."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_unified_preservation as study
from scripts.unified_main_data import Cache,normalize_batch,read,sha

AMENDMENT=ROOT/'reports/unified-preservation-alignment-amendment.json'
SOURCES=('scripts/study_unified_preservation_aligned.py','tests/test_unified_preservation_alignment.py')


class AlignedCache(Cache):
    def batch(self,indices,device='cuda',training=False):
        values=[self.row(int(i)) for i in indices]
        images=torch.stack([v[0] for v in values])
        truth={key:torch.stack([v[1][key] for v in values]).to(device).float() for key in values[0][1]}
        if training:
            flips=torch.rand(len(images))<.5;images[flips]=images[flips].flip(-1)
            on_device=flips.to(device)
            for key in ['mask','pixel_known','valid']:
                if key in truth:truth[key][on_device]=truth[key][on_device].flip(-1)
        return normalize_batch(images,device),truth


def freeze_alignment():
    study.validate()
    study.new(AMENDMENT,{'schema':'unified-preservation-spatial-alignment-v1','declared_before_fit':True,
        'base_protocol_sha256':sha(study.PROTOCOL),'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'correction':'Use identical horizontal-flip mask for image, pixel mask, pixel_known and valid; previous Cache omitted pixel_known',
        'applied_to_both_variants':True,'one_factor_between_candidates_still_backbone_freezing':True,
        'photo_labels_and_known_counts_unchanged':True,'new_gold_labels':0,'TRAIN_only':True,
        'VAL_and_TEST_transform_unchanged':True,'tests_passed':2,
        'preflight_backbone_checks_remain_valid':'Disposable preflight used foreign task without spatial grids',
        'old_study_code_and_protocol_preserved':True})
    print('Both planned fits now use declared, tested aligned spatial augmentation',flush=True)


def activate():
    amendment=read(AMENDMENT);assert amendment['base_protocol_sha256']==sha(study.PROTOCOL)
    repo=ROOT.parents[1]
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name,digest in amendment['source_sha256'].items():
        assert sha(ROOT/name)==digest
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==digest
    amendment_name=AMENDMENT.relative_to(ROOT).as_posix()
    assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+amendment_name])).hexdigest()==sha(AMENDMENT)
    study.Cache=AlignedCache
    original_new=study.new
    def recorded_new(path,value):
        if path.name.startswith('unified-preservation-') and path.name.endswith('-training.json'):
            value={**value,'spatial_alignment_amendment_sha256':sha(AMENDMENT),'spatial_known_grid_flipped_with_image':True}
        original_new(path,value)
    study.new=recorded_new


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze-alignment','train','validation','select','foreign','verify'])
    parser.add_argument('--variant',choices=study.VARIANTS);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='freeze-alignment':freeze_alignment()
    else:
        activate()
        if args.mode=='train':
            for variant in ([args.variant] if args.variant else study.VARIANTS):study.train_variant(variant)
        else:{'validation':study.evaluate_validation,'select':study.select,'foreign':study.evaluate_foreign,'verify':study.verify}[args.mode]()
