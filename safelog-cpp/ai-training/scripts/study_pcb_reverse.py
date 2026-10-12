"""Four actual neural fits, normal-only calibration and a known official benchmark."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from safelog_ai.pcb_reverse import ARCH,SIZE,CATEGORIES,create_teacher,ReverseStudent,loss,anomaly_maps
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc,normal_calibration_threshold
from safelog_ai.retention_distillation import state_sha256
from fetch_equipment_visa import sha
from train_equipment_visa import read,write_new as new,tensors,DATA

RUN = ROOT/'runs/pcb-reverse-v2'
PROTOCOL = ROOT/'reports/pcb-reverse-protocol.json'
LEDGER = ROOT/'runs/pcb-reverse-input-ledger.json'
SOURCES = ('safelog_ai/pcb_reverse.py','scripts/study_pcb_reverse.py','tests/test_pcb_reverse.py')
ORDER = ('pcb3','pcb1','pcb2','pcb4')
EPOCHS,BATCH = 40,32


def sources():
    return {name:sha(ROOT/name) for name in SOURCES}


def snapshot(path):
    stat = path.stat()
    return {'sha256':sha(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def initialize(category,pretrained=True):
    torch.manual_seed(111+CATEGORIES.index(category))
    teacher = create_teacher(pretrained).cuda()
    student = ReverseStudent().cuda()
    return teacher,student


def preflight():
    teacher,student = initialize('pcb3')
    initial,frozen = state_sha256(student),state_sha256(teacher)
    images = torch.randn(2,3,SIZE,SIZE,device='cuda')
    opt = torch.optim.Adam(student.parameters(),lr=.005,betas=(.5,.999))
    with torch.no_grad():
        target = teacher(images)
    predicted = student(target);value = loss(predicted,target)
    assert torch.isfinite(value)
    value.backward();opt.step()
    assert initial!=state_sha256(student) and frozen==state_sha256(teacher)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    teacher.eval();student.eval()
    cpu_teacher,cpu_student = create_teacher(False),ReverseStudent().eval()
    cpu_teacher.load_state_dict({k:v.cpu() for k,v in teacher.state_dict().items()},strict=True)
    cpu_student.load_state_dict({k:v.cpu() for k,v in student.state_dict().items()},strict=True)
    with torch.no_grad():
        gt = teacher(images);gpu = anomaly_maps(student(gt),gt).cpu()
        ct = cpu_teacher(images.cpu());host = anomaly_maps(cpu_student(ct),ct)
    assert torch.allclose(gpu,host,atol=1e-5,rtol=1e-3)
    new(ROOT/'runs/pcb-reverse-preflight.json',{'status':'passed','disposable_optimizer_updates':1,
        'actual_epochs':0,'teacher_state_sha256':frozen,'teacher_frozen':True,'student_changed':True,
        'strict_cpu_gpu_reload_passed':True,'source_sha256':sources(),'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    print('PCB reverse actual GPU update / CPU reload passed',flush=True)


def freeze():
    flight = read(ROOT/'runs/pcb-reverse-preflight.json')
    assert flight['status']=='passed' and flight['source_sha256']==sources()
    manifest = read(DATA/'manifest.json')
    inputs = read(ROOT/'runs/equipment-visa-input-ledger.json')
    for i,(name,record) in enumerate(inputs.items()):
        assert snapshot(ROOT/name)==record,'Changed original PCB input: '+name
        if i%3000==0:
            print({'pcb_input_sha_checked':i,'total':len(inputs)},flush=True)
    recent = read(ROOT/'runs/foreign-calibration-ledger.json')
    inherited = read(ROOT/'runs/conveyor-foreign-expansion-input-ledger.json')
    inherited_records = {**inherited['previous_artifacts'],**inherited['expanded_inputs'],**recent['fresh_inputs']}
    for name,record in inherited_records.items():
        stat = (ROOT/name).stat()
        assert stat.st_size==record['size_bytes'] and stat.st_mtime_ns==record['mtime_ns']
    legacy = {name:sha(path) for folder in ('scripts','tests','safelog_ai') for path in (ROOT/folder).glob('*.py')
              if (name:=path.relative_to(ROOT).as_posix()) not in SOURCES}
    prior = read(ROOT/'reports/foreign-calibration-protocol.json')
    for name,digest in {**prior['legacy_source_sha256'],**prior['source_sha256']}.items():
        assert legacy[name]==digest,name
    artifacts = {}
    for path in (ROOT/'runs').glob('**/final.pt'):
        if 'standalone' not in path.as_posix():
            artifacts[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for pattern in ('equipment-visa*','foreign-calibration*','calibrated-conveyor*','pcb-reverse-preflight*'):
        for path in (ROOT/'reports').glob(pattern):
            if path.is_file():
                artifacts[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(LEDGER,{'pcb_inputs':inputs,'older_model_and_report_artifacts':artifacts,
        'inherited_size_mtime_checked':len(inherited_records),
        'inherited_full_check':'reports/foreign-calibration-verification.json',
        'inherited_check_sha256':sha(ROOT/'reports/foreign-calibration-verification.json')})
    new(PROTOCOL,{'schema':'pcb-reverse-normal-only-v2','declared_before_training':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),'legacy_source_sha256':legacy,
        'manifest_sha256':sha(DATA/'manifest.json'),'input_ledger_sha256':sha(LEDGER),
        'preflight_sha256':sha(ROOT/'runs/pcb-reverse-preflight.json'),
        'method_inspiration':'https://github.com/hq-deng/RD4AD','exact_paper_reproduction':False,
        'dataset':'Official VisA PCB1-4, same unchanged fit/CAL/TEST photos as baseline',
        'source':'https://github.com/amazon-science/spot-diff','license':'CC-BY-4.0',
        'architecture':ARCH,'teacher':'Frozen ImageNet ResNet18 layer1/2/3',
        'student':'448-to128 channel bottleneck; reverse16/32/64 scale residual decoder, no teacher skip',
        'teacher_state_sha256':flight['teacher_state_sha256'],'input_size':SIZE,'batch_size':BATCH,
        'category_order':ORDER,'seeds':{c:111+CATEGORIES.index(c) for c in CATEGORIES},
        'epochs_per_category':EPOCHS,'total_category_model_epochs':EPOCHS*4,
        'optimizer':'Adam beta0.5/0.999','lr':.005,'weight_decay':0.,'gradient_clip_norm':10,
        'scheduler':'Cosine40 eta_min0.00001','loss':'Sum of per-image flattened cosine feature reconstruction loss',
        'score':'Sum3 pixelwise cosine-distance maps, fixed Gaussian sigma4/kernel25; photo maximum',
        'threshold':'Only91 normal CAL/category; ceil((n+1)*0.95) order score, strict >',
        'pixel_threshold':'NormalCAL pooled model256 maps quantile0.995 fixed before TEST',
        'pixel_gt':'Author original anomaly masks, model256 nearest resize; vanishing masks counted',
        'normal_fit_photos':3250,'normal_calibration_photos':364,'test_photos':802,'counts':manifest['counts'],
        'normal_only_neural_training':True,'source_test_used_for_fit_or_calibration':False,
        'source_test_is_known_reused_benchmark':True,'independent_factory_or_physical_boards_verified':False,
        'final_checkpoint':'Final40 fixed; no test-based epoch or threshold selection',
        'baseline':'STFPM final20 per category; different structure, aggregation and update budget',
        'same_optimizer_update_budget_as_baseline':False,'mechanical_failure_diagnosis':False,
        'new_ai_or_human_labels':0,'app_default_promoted':False,'tests_expected':4,
        'disposable_preflight_updates':1,'all_eight_photo_rates_target':.05})
    print({'protocol':'frozen','categories':ORDER,'actual_training_target_epochs_each':40,'legacy_sources':len(legacy)},flush=True)


def validate():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources() and protocol['manifest_sha256']==sha(DATA/'manifest.json')
    assert protocol['input_ledger_sha256']==sha(LEDGER)
    return protocol


def committed():
    repo = ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        payload = subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])
        assert hashlib.sha256(payload).hexdigest()==sha(ROOT/name),'Commit frozen settings before train'
    return subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()


def train():
    protocol,commit = validate(),committed()
    assert not RUN.exists();RUN.mkdir()
    manifest = read(DATA/'manifest.json');points=[];started=time.monotonic()
    for category in ORDER:
        rows = [r for r in manifest['rows'] if r['object']==category and r['phase']=='fit']
        cal_rows = [r for r in manifest['rows'] if r['object']==category and r['phase']=='calibration']
        assert all(r['split']=='train' and r['label']=='normal' for r in rows+cal_rows)
        images,cal_images = tensors(rows),tensors(cal_rows)
        teacher,student = initialize(category)
        initial,frozen = state_sha256(student),state_sha256(teacher)
        assert frozen==protocol['teacher_state_sha256']
        opt = torch.optim.Adam(student.parameters(),lr=.005,betas=(.5,.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.00001)
        directory = RUN/category;directory.mkdir();history=[];updates=0
        for epoch in range(1,EPOCHS+1):
            student.train();total=0.
            for indices in torch.randperm(len(images)).split(BATCH):
                with torch.no_grad():
                    target = teacher(images[indices].cuda())
                opt.zero_grad(set_to_none=True)
                value = loss(student(target),target);assert torch.isfinite(value)
                value.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),10,error_if_nonfinite=True);opt.step()
                total += float(value.detach())*len(indices);updates += 1
            scheduler.step()
            entry = {'category':category,'epoch':epoch,'train_loss':total/len(images),'optimizer_updates':updates,
                     'elapsed_minutes':(time.monotonic()-started)/60}
            new(directory/f'epoch-{epoch:02d}.json',entry);history.append(entry)
            print(json.dumps(entry),flush=True)
        student.eval();assert state_sha256(student)!=initial and state_sha256(teacher)==frozen
        assert all(p.grad is None for p in teacher.parameters())
        cal_maps=[]
        with torch.inference_mode():
            for images_batch in cal_images.split(BATCH):
                target=teacher(images_batch.cuda());cal_maps.append(anomaly_maps(student(target),target).cpu())
        maps=torch.cat(cal_maps);scores=maps.flatten(1).amax(1).double().tolist()
        threshold=normal_calibration_threshold(scores);pixel_threshold=float(torch.quantile(maps.flatten(),.995))
        metadata={'architecture':ARCH,'equipment_category':category,'input_size':SIZE,'epochs':EPOCHS,
            'train_photos':len(rows),'calibration_photos':len(cal_rows),'optimizer_updates':updates,
            'threshold':threshold,'pixel_threshold':pixel_threshold,'initial_state_sha256':initial,
            'final_student_sha256':state_sha256(student),'teacher_sha256':frozen,
            'manifest_sha256':sha(DATA/'manifest.json'),'protocol_sha256':sha(PROTOCOL),'frozen_before_fit_commit':commit,
            'source_test_used_for_fit_or_calibration':False,'mechanical_failure_state':'not_assessed'}
        torch.save({**metadata,'student':{k:v.cpu() for k,v in student.state_dict().items()},
                    'teacher':{k:v.cpu() for k,v in teacher.state_dict().items()}},directory/'final.pt')
        cpu_t,cpu_s=create_teacher(False),ReverseStudent().eval()
        cpu_t.load_state_dict(torch.load(directory/'final.pt',map_location='cpu',weights_only=True)['teacher'],strict=True)
        cpu_s.load_state_dict(torch.load(directory/'final.pt',map_location='cpu',weights_only=True)['student'],strict=True)
        with torch.inference_mode():
            gt=teacher(cal_images[:2].cuda());gpu=anomaly_maps(student(gt),gt).cpu()
            ct=cpu_t(cal_images[:2]);host=anomaly_maps(cpu_s(ct),ct)
        assert torch.allclose(gpu,host,atol=1e-5,rtol=1e-3)
        new(directory/'history.json',history)
        new(directory/'calibration.json',{'image_ids':[r['image'] for r in cal_rows],'scores':scores,
            'threshold':threshold,'pixel_threshold':pixel_threshold,'normal_only':True,'source_test_used':False})
        points.append({**metadata,'weights_sha256':sha(directory/'final.pt'),'strict_cpu_gpu_reload':True})
        del teacher,student,opt,cpu_t,cpu_s,images,cal_images,maps,cal_maps
        torch.cuda.empty_cache()
    new(ROOT/'reports/pcb-reverse-training.json',{'status':'all_four_trained_and_calibrated_before_test',
        'categories':points,'actual_category_model_epochs':160,'actual_optimizer_updates':sum(p['optimizer_updates'] for p in points),
        'elapsed_training_minutes':(time.monotonic()-started)/60,'source_test_used_for_fit_or_calibration':False,
        'known_test_not_scored_by_new_models_yet':True,'protocol_sha256':sha(PROTOCOL)})
    print('Four final40 models and normalCAL thresholds frozen before benchmark',flush=True)


def evaluate():
    validate();training=read(ROOT/'reports/pcb-reverse-training.json')
    assert not (ROOT/'reports/pcb-reverse-results.json').exists()
    results=[];manifest=read(DATA/'manifest.json')
    for entry in training['categories']:
        category=entry['equipment_category'];path=RUN/category/'final.pt';assert sha(path)==entry['weights_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        teacher,student=initialize(category,False);student.eval()
        teacher.load_state_dict(state['teacher'],strict=True);student.load_state_dict(state['student'],strict=True)
        rows=[r for r in manifest['rows'] if r['object']==category and r['phase']=='test'];images=tensors(rows)
        all_maps=[]
        with torch.inference_mode():
            for batch in images.split(BATCH):
                target=teacher(batch.cuda());all_maps.append(anomaly_maps(student(target),target).cpu())
        maps=torch.cat(all_maps);scores=maps.flatten(1).amax(1).double().tolist();labels=[int(r['label']=='anomaly') for r in rows]
        rates=classification_rates(labels,scores,entry['threshold']);pixels=dict(tp=0,fp=0,tn=0,fn=0);vanished=0
        for row,heatmap in zip(rows,maps):
            mask=np.zeros((SIZE,SIZE),dtype=bool)
            if row['mask']:
                with Image.open(DATA/'originals'/row['mask']) as image:
                    native=np.array(image.convert('L'))>0
                    mask=np.array(image.convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
                vanished += int(native.any() and not mask.any())
            pred=heatmap.numpy()>entry['pixel_threshold']
            for key,value in [('tp',pred&mask),('fp',pred&~mask),('tn',~pred&~mask),('fn',~pred&mask)]:
                pixels[key]+=int(value.sum())
        denom=pixels['tp']+pixels['fp']+pixels['fn']
        results.append({'category':category,**rates,'auroc':rank_auroc(labels,scores),'pixel_counts_model256':pixels,
            'pixel_iou_model256':pixels['tp']/denom if denom else None,'native_positive_mask_vanished_at256':vanished,
            'threshold':entry['threshold'],'pixel_threshold':entry['pixel_threshold'],'weights_sha256':entry['weights_sha256']})
        new(RUN/category/'test-predictions.json',{'image_ids':[r['image'] for r in rows],'labels':labels,'scores':scores})
        torch.save(maps,RUN/category/'test-maps.pt')
    pooled={key:sum(r[key] for r in results) for key in ('tp','fp','tn','fn')}
    pooled.update(fpr=pooled['fp']/(pooled['fp']+pooled['tn']),fnr=pooled['fn']/(pooled['fn']+pooled['tp']))
    new(ROOT/'reports/pcb-reverse-results.json',{'status':'evaluated','categories':results,'pooled':pooled,
        'all_eight_photo_rates_below5_percent':all(r['fpr']<.05 and r['fnr']<.05 for r in results),
        'normal_only_thresholds_fixed_before_test':True,'source_test_used_for_fit_or_calibration':False,
        'known_official_test_reused':True,'independent_factory_or_physical_boards_verified':False,
        'mechanical_failure_diagnosis':False,'app_default_promoted':False,'protocol_sha256':sha(PROTOCOL)})
    print(json.dumps({'categories':results,'pooled':pooled}),flush=True)


def verify():
    protocol=validate();ledger=read(LEDGER)
    for name,digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    for key in ('pcb_inputs','older_model_and_report_artifacts'):
        for name,record in ledger[key].items():
            assert snapshot(ROOT/name)==record,name
    training=read(ROOT/'reports/pcb-reverse-training.json');result=read(ROOT/'reports/pcb-reverse-results.json')
    assert len(training['categories'])==4 and training['actual_category_model_epochs']==160
    for entry in training['categories']:
        category=entry['equipment_category'];history=read(RUN/category/'history.json')
        assert [h['epoch'] for h in history]==list(range(1,41))
        assert entry['optimizer_updates']==history[-1]['optimizer_updates']==((entry['train_photos']+31)//32)*40
        assert entry['strict_cpu_gpu_reload'] and sha(RUN/category/'final.pt')==entry['weights_sha256']
        pred=read(RUN/category/'test-predictions.json');rates=classification_rates(pred['labels'],pred['scores'],entry['threshold'])
        row=next(r for r in result['categories'] if r['category']==category)
        assert all(row[k]==v for k,v in rates.items())
    new(ROOT/'reports/pcb-reverse-verification.json',{'status':'passed','actual_category_model_epochs':160,
        'actual_optimizer_updates':training['actual_optimizer_updates'],'tests_passed':4,'preflight_updates':1,
        'strict_cpu_gpu_reload_all_four':True,'teacher_unchanged_all_four':True,
        'pcb_input_sha_size_mtime_preserved':len(ledger['pcb_inputs']),
        'older_model_artifacts_sha_size_mtime_preserved':len(ledger['older_model_and_report_artifacts']),
        'legacy_sources_sha_preserved':len(protocol['legacy_source_sha256']),
        'inherited_other_inputs_size_mtime_checked':ledger['inherited_size_mtime_checked'],
        'known_official_test_reused':True,'app_default_promoted':False,'protocol_sha256':sha(PROTOCOL)})
    print('All four actual neural fits, immutable inputs and benchmark counts verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','freeze','train','evaluate','verify'))
    args=parser.parse_args();torch.set_num_threads(4);os.environ['TORCH_HOME']=str(ROOT/'.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'preflight':preflight,'freeze':freeze,'train':train,'evaluate':evaluate,'verify':verify}[args.mode]()
