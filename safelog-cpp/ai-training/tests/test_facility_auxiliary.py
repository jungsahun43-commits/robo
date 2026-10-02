import unittest
import torch
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from safelog_ai.auxiliary_classifier import ARCH,AUX_CLASSES
from safelog_ai.spatial_classifier import ARCH as BASE_ARCH
from safelog_ai.presence_classifier import build_model
from scripts.prepare_facility_auxiliary import original_targets
from scripts.train_facility_target import masked_focal
from scripts.train_facility_spatial import SpatialPhotos


class AuxiliaryTests(unittest.TestCase):
    def test_base_initializer_keeps_same_seven_predictions_before_finetuning(self):
        torch.set_num_threads(2)
        base=build_model(7,architecture=BASE_ARCH).eval();aux=build_model(7,architecture=ARCH).eval()
        mismatch=aux.load_state_dict(base.state_dict(),strict=False)
        self.assertEqual(set(mismatch.missing_keys),{'auxiliary_head.weight','auxiliary_head.bias'})
        self.assertEqual(mismatch.unexpected_keys,[])
        image=torch.randn(1,3,128,128)
        with torch.inference_mode():original=base(image);extended=aux(image)
        self.assertTrue(torch.equal(original,extended))

    def test_parent_tags_do_not_transfer_to_detail_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'crop.png';mask=root/'mask.npz'
            Image.new('RGB',(16,16)).save(image)
            np.savez_compressed(mask,mask=np.zeros((7,80,80),dtype=np.uint8),known=np.ones(7,dtype=np.uint8))
            item={'image':str(image),'parent_image':'parent.jpg','domain':'dacl','targets':[0]*7,'pixel_target':str(mask)}
            crop=SpatialPhotos([item],{'parent.jpg':[1]*19})[0]
            self.assertEqual(crop[7].sum().item(),0.)
            full=SpatialPhotos([item],{str(image):[1]*19})[0]
            self.assertEqual(full[7].sum().item(),19.)

    def test_original_nuisance_tag_does_not_become_spalling(self):
        targets=original_targets({'shapes':[{'label':'Rockpocket'},{'label':'ExposedRebars'}]})
        self.assertEqual(targets[AUX_CLASSES.index('Spalling')],0)
        self.assertEqual(targets[AUX_CLASSES.index('Rockpocket')],1)
        with self.assertRaises(ValueError):original_targets({'shapes':[{'label':'new-unknown'}]})

    def test_unknown_auxiliary_photo_has_zero_gradient(self):
        scores=torch.zeros(2,19,requires_grad=True)
        targets=torch.zeros_like(scores);known=torch.zeros_like(scores);known[0]=1
        masked_focal(scores,targets,known,torch.ones(19),1.).backward()
        self.assertEqual(scores.grad[1].abs().sum().item(),0.)
        self.assertGreater(scores.grad[0].abs().sum().item(),0.)

    def test_seven_public_logits_and_nineteen_training_only_logits(self):
        torch.set_num_threads(2);model=build_model(7,architecture=ARCH).eval()
        with torch.inference_mode():photo,maps,aux=model.forward_training(torch.zeros(2,3,128,128))
        self.assertEqual(tuple(photo.shape),(2,7));self.assertEqual(tuple(aux.shape),(2,19))
        self.assertEqual(tuple(maps.shape),(2,7,16,16))
        with torch.inference_mode():public=model(torch.zeros(2,3,128,128))
        self.assertEqual(tuple(public.shape),(2,7))
