"""Guard alignment, unknown supervision and the legacy export boundary."""
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts import facility_dense_auxiliary as study
from scripts import train_facility_context as original
from scripts.facility_resolution_study import ResolutionPhotos
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.dense_auxiliary_classifier import DenseAuxiliaryClassifier, original_state


class DensePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def dataset(self, directory):
        root=Path(directory); rgb=np.zeros((96,160,3),np.uint8); rgb[:,:80,0]=200; rgb[:,80:,1]=100
        Image.fromarray(rgb).save(root/"photo.png")
        mask=np.zeros((7,80,80),np.uint8); mask[0,:,:20]=1
        np.savez(root/"old.npz",mask=mask,known=np.ones(7,np.uint8))
        dense=np.zeros((19,80,80),np.uint8); dense[0,:,:20]=1; dense[1,:,:20]=1
        np.savez(root/"dense.npz",masks=dense,known=np.ones(19,np.uint8))
        items=[{"image":"photo.png","pixel_target":"old.npz","domain":"dacl","targets":[1,0,-1,0,0,0,0]},
               {"image":"photo.png","pixel_target":"old.npz","domain":"damsegment","targets":[1,0,-1,-1,-1,-1,-1]},
               {"image":"photo.png","pixel_target":"old.npz","domain":"dacl","targets":[1,0,-1,0,0,0,0]}]
        return {"items":items,"auxiliary":{"photo.png":[1,1]+[0]*17},"full_count":2,
                "dense":{"photo.png":{"mask":"dense.npz"}}}

    def test_original_contract_and_rng_match(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study,"ROOT",Path(directory)), patch.object(original,"ROOT",Path(directory)):
            data=self.dataset(directory); candidate=study.DensePhotos(data)
            control=ResolutionPhotos(data["items"],data["auxiliary"],2,study.DOMAINS,640)
            for seed in (1,2,17):
                random.seed(seed); torch.manual_seed(seed); reference=control[0]
                python_state,torch_state=random.getstate(),torch.get_rng_state().clone()
                random.seed(seed); torch.manual_seed(seed); treatment=candidate[0]
                self.assertEqual(random.getstate(),python_state); self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
                for x,y in zip(reference,treatment[:10]):
                    if isinstance(x,torch.Tensor): self.assertTrue(torch.equal(x,y))
                    else:self.assertEqual(x,y)

    def test_same_flip_for_both_masks_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study,"ROOT",Path(directory)), patch.object(study.random,"random",return_value=0.):
            dataset=study.DensePhotos(self.dataset(directory)); row=dataset[0]
            self.assertTrue(torch.equal(row[4][0],row[10][0])); self.assertTrue(torch.equal(row[10][0],row[10][1]))
            self.assertEqual(int(row[10][0,:,:20].sum()),0); self.assertEqual(int(row[10][0,:,-20:].sum()),1600)

    def test_other_sources_and_crops_stay_unknown(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study,"ROOT",Path(directory)):
            dataset=study.DensePhotos(self.dataset(directory))
            for index in (1,2):
                row=dataset[index]; self.assertEqual(int(row[11].count_nonzero()),0)
                self.assertEqual(int(row[10].count_nonzero()),0); self.assertEqual(row[12],0)

    def test_unknown_dense_loss_and_gradient_zero(self):
        logits=torch.randn(2,19,8,8,requires_grad=True); targets=torch.ones_like(logits)
        loss=spatial_loss(logits,targets,torch.zeros(2,19),torch.ones(19))
        self.assertEqual(float(loss.detach()),0.); loss.backward(); self.assertEqual(int(logits.grad.count_nonzero()),0)

    def test_export_after_actual_dense_update_preserves_predictions(self):
        model=DenseAuxiliaryClassifier(7,pretrained=False).eval(); inputs=torch.randn(2,3,128,128)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for _ in range(2):
            optimizer.zero_grad(); outputs=model.forward_dense_training(inputs)
            (outputs[0].square().mean()+outputs[3].sub(1).square().mean()).backward(); optimizer.step()
        legacy=AuxiliaryClassifier(7,pretrained=False).eval(); legacy.load_state_dict(original_state(model),strict=True)
        with torch.no_grad(): self.assertTrue(torch.equal(model(inputs),legacy(inputs)))
        self.assertEqual(len(legacy.state_dict()),324); self.assertEqual(len(model.state_dict()),328)


if __name__=="__main__":unittest.main()
