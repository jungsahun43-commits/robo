"""TRAIN-only disposable AMP checks for the frozen 640/960 input comparison."""
from pathlib import Path
import hashlib
from io import BytesIO
import json
import platform
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.resolution_classifier import ResolutionClassifier
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.presence_classifier import PresenceClassifier, MEAN, STD
from scripts.facility_resolution_study import ResolutionPhotos, prepare_data, validate_protocol
from scripts.preflight_facility_detail import select_train_replay, seed_model, batch_fingerprint
from scripts.train_facility_target import read, save, sha, masked_focal, TARGETS, DOMAINS

PROTOCOL = ROOT / 'reports/facility-resolution-study-protocol.json'
OUTPUT = ROOT / 'runs/facility-resolution-preflight.json'


def replay(items, auxiliary, imgsz, seed):
    seed_model(seed + 2)
    sampler = WeightedRandomSampler(torch.ones(len(items), dtype=torch.float64), 24,
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(ResolutionPhotos(items, auxiliary, len(items), DOMAINS, imgsz=imgsz),
                        batch_size=8, sampler=sampler, num_workers=4, persistent_workers=True,
                        pin_memory=True, generator=torch.Generator().manual_seed(seed+1))
    iterator = iter(loader); rows = []; first = None
    try:
        for _ in range(3):
            batch = next(iterator)
            if first is None: first = batch
            rows.append({'non_image_batch_sha256': batch_fingerprint(batch[1:]),
                         'image_tensor_sha256': batch_fingerprint(batch[:1]),
                         'indices_sha256': hashlib.sha256(batch[-1].numpy().astype('<i8').tobytes()).hexdigest()})
    finally:
        iterator._shutdown_workers()
    return rows, first


def run():
    if OUTPUT.exists(): raise ValueError('Preserve the prior preflight evidence')
    if not torch.cuda.is_available(): raise ValueError('Actual CUDA AMP backward is required')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    protocol = validate_protocol(read(PROTOCOL), ROOT)
    data = prepare_data(ROOT, protocol)
    items, auxiliary = select_train_replay(data['manifest'], data['auxiliary_manifest'])
    initial = ROOT / 'runs' / protocol['reference'] / 'best.pt'
    paths = {'initial': initial, 'core': ROOT / 'data/facility-spatial-training/train.json',
             'auxiliary': ROOT / 'data/facility-auxiliary-training/train.json',
             'app_profile': ROOT / 'reports/facility-inference-profile.json', 'protocol': PROTOCOL}
    before = {key: sha(path) for key, path in paths.items()}
    checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
    models, rng, augmented, batches = {}, {}, {}, {}
    for variant, constructor in [('control', AuxiliaryClassifier), ('highres', ResolutionClassifier)]:
        seed_model(protocol['seed'])
        model = constructor(7, pretrained=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        rng[variant] = torch.get_rng_state().clone()
        models[variant] = model
        augmented[variant], batches[variant] = replay(items, auxiliary,
                                  protocol['imgsz_by_variant'][variant], protocol['seed'])
    if not torch.equal(rng['control'], rng['highres']): raise ValueError('Constructor RNG differs')
    for left, right in zip(augmented['control'], augmented['highres']):
        if any(left[key] != right[key] for key in ('non_image_batch_sha256', 'indices_sha256')):
            raise ValueError('TRAIN mask/labels/known/indices differ across the resolution pair')
    outputs = []
    for model in models.values():
        model.to('cuda').eval()
        with torch.inference_mode(): outputs.append(tuple(v.cpu() for v in model.forward_training(batches['control'][0].to('cuda'))))
    if not all(torch.equal(a,b) for a,b in zip(*outputs)):
        raise ValueError('640 initial outputs must match tensor-for-tensor')
    del outputs
    photo = data['sampling_data']['photo_weights'].to('cuda')
    pixel = data['supervision_weights']['pixel_weights'].to('cuda')
    tags = data['supervision_weights']['auxiliary_weights'].to('cuda')
    emphasis = torch.ones(7, device='cuda')
    for label in TARGETS: emphasis[data['classes'].index(label)] = 2.
    diagnostics = {}
    for variant, model in models.items():
        batch = batches[variant]; raw_shapes = []
        hook = model.segmentation_head.register_forward_hook(lambda module, args, output: raw_shapes.append(list(output.shape)))
        model.eval()
        with torch.inference_mode(): values = model.forward_training(batch[0].to('cuda'))
        hook.remove()
        expected_raw = 80 if variant == 'control' else 120
        if (raw_shapes != [[8,7,expected_raw,expected_raw]]
                or [list(v.shape) for v in values] != [[8,7],[8,7,80,80],[8,19]]
                or not all(bool(torch.isfinite(v).all()) for v in values)):
            raise ValueError('Resolution raw/projected/public output contract failed')
        del values
        # Reuse the same TRAIN images to check the actual epoch-VAL batch size.
        # This is a memory/shape test, not a validation-data prediction.
        with torch.inference_mode():
            repeated_values = model.forward_training(batch[0].repeat(2,1,1,1).to('cuda'))
        if ([list(v.shape) for v in repeated_values] != [[16,7],[16,7,80,80],[16,19]]
                or not all(bool(torch.isfinite(v).all()) for v in repeated_values)):
            raise ValueError('The validation-size TRAIN replay forward failed')
        del repeated_values
        model.train(); torch.cuda.reset_peak_memory_stats()
        backbone = list(model.backbone.parameters()); ids = {id(p) for p in backbone}
        optimizer = torch.optim.AdamW([{'params':backbone,'lr':protocol['backbone_lr']},
                  {'params':[p for p in model.parameters() if id(p) not in ids],'lr':protocol['head_lr']}], weight_decay=.0002)
        steps = []; handle = optimizer.register_step_post_hook(lambda *args: steps.append(1))
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            scores, maps, aux = model.forward_training(batch[0].to('cuda'))
            loss = masked_focal(scores,batch[1].to('cuda'),batch[2].to('cuda'),photo[batch[3]],1.,emphasis)
            loss += spatial_loss(maps,batch[4].to('cuda'),batch[5].to('cuda'),pixel)
            loss += protocol['auxiliary_weight']*masked_focal(aux,batch[6].to('cuda'),batch[7].to('cuda'),tags,1.)
        if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite disposable loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('Disconnected or nonfinite gradients')
        scaler.step(optimizer); scaler.update(); handle.remove(); torch.cuda.synchronize()
        if steps != [1] or not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError('Disposable AMP update failed')
        state = {key:value.detach().cpu() for key,value in model.state_dict().items()}
        serialized = BytesIO()
        torch.save({'architecture':protocol['architecture_by_variant'][variant],
                    'state_dict':state,'classes':protocol['classes'],
                    'imgsz':protocol['imgsz_by_variant'][variant],'mean':MEAN,'std':STD},serialized)
        checkpoint_digest = hashlib.sha256(serialized.getvalue()).hexdigest()
        serialized.seek(0)
        restored = PresenceClassifier(serialized,device='cpu')
        if (restored.imgsz != protocol['imgsz_by_variant'][variant]
                or not all(torch.equal(value,state[key]) for key,value in restored.model.state_dict().items())):
            raise ValueError('Disposable factory serialization changed state or input size')
        with torch.inference_mode(): restored_score = restored.model(batch[0][:1])
        if list(restored_score.shape) != [1,7] or not bool(torch.isfinite(restored_score).all()):
            raise ValueError('CPU factory restored output failed')
        diagnostics[variant] = {'imgsz':protocol['imgsz_by_variant'][variant],
             'raw_map_shape':raw_shapes[0], 'loss_and_pool_grid':[80,80], 'photo_shape':[8,7],
             'auxiliary_shape':[8,19], 'loss':float(loss.detach()), 'actual_optimizer_steps':len(steps),
             'validation_like_train_repeated_batch_size':16, 'validation_size_forward_finite':True,
             'disposable_serialization_and_factory_cpu_reload_verified':True,
             'disposable_serialized_checkpoint_sha256':checkpoint_digest,
             'all_gradients_finite':True, 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
             'parameter_count':sum(p.numel() for p in model.parameters())}
        del scores, maps, aux, loss, gradients, optimizer, scaler, restored, restored_score, state, serialized
    if diagnostics['control']['parameter_count'] != diagnostics['highres']['parameter_count']:
        raise ValueError('This input study cannot add parameters')
    if before != {key:sha(path) for key,path in paths.items()}:
        raise ValueError('Disposable integration check changed evidence or app profile')
    result = {'schema':'facility_resolution_preflight_v1','status':'passed','training_epochs':0,
              'accuracy_measured':False,'actual_train_source_cases':len(items),
              'bounded_replay_batches_per_variant':3,'constructor_rng_equal':True,
              '640_initial_outputs_identical':True,'mask_label_known_index_replay_equal':True,
              'resized_image_tensors_expected_to_differ':True,'replay':augmented,
              'variants':diagnostics,'source_sha256':protocol['source_sha256'],
              'protected_file_sha256':before,'protected_files_unchanged':True,
              'python':platform.python_version(),'torch':torch.__version__,
              'gpu':torch.cuda.get_device_name(0),'protocol_sha256':sha(PROTOCOL),
              'scope':'Eight TRAIN cases, bounded replay and disposable updates; no full epoch equivalence or accuracy claim'}
    save(OUTPUT,result)
    print(json.dumps({'status':'passed','train_cases':len(items),'variants':diagnostics},indent=2))


if __name__ == '__main__': run()
