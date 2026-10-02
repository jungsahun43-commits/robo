"""Original DACL tags regularize training; public evidence stays seven classes."""
import torch
from .spatial_classifier import SpatialClassifier

ARCH='lraspp_mobilenet_facility_auxiliary_v1'
AUX_CLASSES=('ACrack','Bearing','Cavity','Crack','Drainage','EJoint','Efflorescence',
             'ExposedRebars','Graffiti','Hollowareas','JTape','PEquipment','Restformwork',
             'Rockpocket','Rust','Spalling','WConccor','Weathering','Wetspot')


class AuxiliaryClassifier(SpatialClassifier):
    def __init__(self,classes=7,pretrained=False):
        super().__init__(classes,pretrained)
        self.auxiliary_head=torch.nn.Linear(960,len(AUX_CLASSES))

    def forward_training(self,image):
        features=self.backbone(image);maps=self.segmentation_head(features)
        pooled=features['high'].mean((-2,-1))
        global_logits=self.photo_head(pooled)
        spatial_logits=maps.flatten(2).topk(32,dim=-1).values.mean(-1)
        mixture=self.mix.sigmoid()[None,:]
        return global_logits*(1-mixture)+spatial_logits*mixture,maps,self.auxiliary_head(pooled)

    def forward_details(self,image):
        photo,maps,_=self.forward_training(image)
        return photo,maps
