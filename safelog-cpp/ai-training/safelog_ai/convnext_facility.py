"""Trainable ImageNet ConvNeXt-Tiny + facility FPN, public seven outputs.

This is an architecture/initialization package, not a MobileNet-state transfer.
Only explicit local verified ImageNet weights initialize the feature extractor.
LayerNorm/GroupNorm require no running BatchNorm statistics. During training,
non-reentrant block checkpointing preserves stochastic-depth RNG while saving
memory. Official ImageNet classifier logits are discarded.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import convnext_tiny
from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.presence_classifier import image_transform
from safelog_ai.retention_distillation import state_sha256

ARCH="convnext_tiny_facility_fpn_auxiliary_v1"
OFFICIAL_SHA="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
CLASSES=("concrete_crack","concrete_spalling","rust_stain","exposed_rebar","wet_surface","efflorescence","surface_cavity")

def require(condition,message):
    if not condition:raise ValueError(message)

class ConvnextFacility(nn.Module):
    def __init__(self,classes=7,pretrained=False,activation_checkpointing=False):
        super().__init__();require(classes==7 and pretrained is False,"Offline seven-class model required")
        require(type(activation_checkpointing)is bool,"Checkpointing must be boolean")
        source=convnext_tiny(weights=None,stochastic_depth_prob=.1)
        self.backbone=source.features;self.pool_norm=source.classifier[0]
        self.low_projection=nn.Sequential(nn.Conv2d(192,128,1),nn.GroupNorm(8,128),nn.ReLU())
        self.high_projection=nn.Sequential(nn.Conv2d(768,128,1),nn.GroupNorm(8,128),nn.ReLU())
        self.map_head=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU(),nn.Conv2d(128,7,1))
        self.photo_head=nn.Linear(768,7);self.auxiliary_head=nn.Linear(768,len(AUX_CLASSES))
        self.mix=nn.Parameter(torch.zeros(7));self.activation_checkpointing=activation_checkpointing

    def encode(self,image):
        require(image.ndim==4 and image.shape[1]==3 and image.shape[-2]>=64 and image.shape[-1]>=64
            and image.shape[-2]%32==image.shape[-1]%32==0,"Stride32-compatible RGB images >=64 required")
        value=image;low=None;active=self.activation_checkpointing and self.training and torch.is_grad_enabled()
        for index,stage in enumerate(self.backbone):
            if active and index in(1,3,5,7):
                for block in stage:value=checkpoint(block,value,use_reentrant=False,preserve_rng_state=True)
            elif active:value=checkpoint(stage,value,use_reentrant=False,preserve_rng_state=True)
            else:value=stage(value)
            if index==3:low=value
        pooled=self.pool_norm(value.mean((-2,-1),keepdim=True)).flatten(1)
        return low,value,pooled

    def forward_training(self,image):
        low,high,pooled=self.encode(image)
        local=self.low_projection(low);context=self.high_projection(high)
        context=F.interpolate(context,size=local.shape[-2:],mode="bilinear",align_corners=False)
        maps=self.map_head(torch.cat((local,context),dim=1))
        spatial=maps.flatten(2).topk(min(32,maps.shape[-2]*maps.shape[-1]),dim=-1).values.mean(-1)
        mixture=self.mix.sigmoid()[None,:]
        photo=self.photo_head(pooled)*(1-mixture)+spatial*mixture
        return photo,maps,self.auxiliary_head(pooled)

    def forward(self,image):return self.forward_training(image)[0]
    def forward_details(self,image):
        photo,maps,_=self.forward_training(image);return photo,maps

    def encoder_state(self):
        return {k:v for k,v in self.state_dict().items()if k.startswith(("backbone.","pool_norm."))}

def load_imagenet(model,path):
    import hashlib
    path=Path(path)
    with path.open("rb")as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
    require(digest==OFFICIAL_SHA,"Verified official ImageNet bytes required")
    original=torch.load(path,map_location="cpu",weights_only=True);current=model.encoder_state();mapped={}
    for name,target in current.items():
        key="features."+name[len("backbone."):]if name.startswith("backbone.")else"classifier.0."+name[len("pool_norm."):]
        value=original.get(key)
        require(isinstance(value,torch.Tensor)and value.shape==target.shape and value.dtype==target.dtype
            and torch.isfinite(value).all(),"Official encoder tensor differs: "+key)
        mapped[name]=value
    used={"features."+k[len("backbone."):]if k.startswith("backbone.")else"classifier.0."+k[len("pool_norm."):]for k in current}
    require(set(original)==used|{"classifier.2.weight","classifier.2.bias"}and len(mapped)==180,"Official inventory differs")
    merged=model.state_dict();merged.update(mapped);model.load_state_dict(merged,strict=True)
    require(all(torch.equal(model.state_dict()[k],v)for k,v in mapped.items()),"Official transfer is not exact")
    require(all(p.requires_grad for p in model.backbone.parameters())and all(p.requires_grad for p in model.pool_norm.parameters()),"New encoder must be trainable")
    return {"weights_sha256":digest,"official_encoder_tensors_transferred":len(mapped),"encoder_state_sha256":state_sha256(model.encoder_state()),
        "official_classifier_discarded":True,"all_feature_extractor_parameters_trainable":True,"original_mobilenet_state_transferred":False}

def inventory(model):
    return {"parameter_count":sum(p.numel()for p in model.parameters()),"trainable_parameter_count":sum(p.numel()for p in model.parameters()if p.requires_grad),
        "state_tensor_count":len(model.state_dict()),"encoder_state_tensor_count":len(model.encoder_state()),
        "student_batchnorm_modules":sum(isinstance(m,nn.modules.batchnorm._BatchNorm)for m in model.modules()),
        "low_channels":192,"high_channels":768,"native_map_size_for_640":[80,80],"stochastic_depth_probability":.1}

class ConvnextResearchPresence:
    """Strict private-checkpoint adapter with the existing evaluator interface."""
    def __init__(self,weights,device="cpu"):
        record=torch.load(weights,map_location="cpu",weights_only=True)
        require(record.get("architecture")==ARCH and record.get("classes")==list(CLASSES)
            and record.get("auxiliary_classes")==list(AUX_CLASSES)and record.get("imgsz")==640,"Research metadata differs")
        self.classes=list(CLASSES);self.transform=image_transform(640);self.model=ConvnextFacility()
        self.model.load_state_dict(record["state_dict"],strict=True);self.model.to(device).eval();self.device=device
