"""New encoder contracts, checkpoint gradients and strict research loading."""
import tempfile
from pathlib import Path
import unittest
import torch
from safelog_ai.convnext_facility import ConvnextFacility,ConvnextResearchPresence,ARCH,CLASSES,inventory,load_imagenet
from safelog_ai.auxiliary_classifier import AUX_CLASSES

class ConvnextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(56)
        cls.model=ConvnextFacility();cls.model.eval();cls.input=torch.randn(2,3,64,64)
    def test_three_output_shapes(self):
        with torch.no_grad():out=self.model.forward_training(self.input)
        self.assertEqual([list(x.shape)for x in out],[[2,7],[2,7,8,8],[2,19]])
        self.assertTrue(all(torch.isfinite(x).all()for x in out))
    def test_public_matches_photo_output(self):
        with torch.no_grad():a=self.model(self.input);b=self.model.forward_training(self.input)[0]
        self.assertTrue(torch.equal(a,b))
    def test_no_student_batchnorm_or_frozen_parameters(self):
        inv=inventory(self.model);self.assertEqual(inv['student_batchnorm_modules'],0)
        self.assertEqual(inv['parameter_count'],inv['trainable_parameter_count']);self.assertGreater(inv['parameter_count'],27000000)
    def test_trainable_official_transfer(self):
        root=Path(__file__).resolve().parents[1]
        proof=load_imagenet(self.model,root/'data/pretrained/convnext-tiny-imagenet1k-v1.pth')
        self.assertEqual(proof['official_encoder_tensors_transferred'],180);self.assertFalse(proof['original_mobilenet_state_transferred'])
    def test_invalid_spatial_shape_rejected(self):
        with self.assertRaises(ValueError):self.model(torch.zeros(1,3,65,64))
    def test_checkpointed_and_direct_eval_equal(self):
        self.model.activation_checkpointing=True
        with torch.no_grad():a=self.model.forward_training(self.input)
        self.model.activation_checkpointing=False
        with torch.no_grad():b=self.model.forward_training(self.input)
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(a,b)))
    def test_checkpoint_training_reaches_early_and_late_blocks(self):
        self.model.train();self.model.activation_checkpointing=True;self.model.zero_grad(set_to_none=True)
        torch.manual_seed(71);out=self.model.forward_training(self.input)
        sum(x.square().mean()for x in out).backward()
        for parameter in(self.model.backbone[0][0].weight,self.model.backbone[7][-1].block[0].weight,self.model.map_head[-1].weight,self.model.photo_head.weight,self.model.auxiliary_head.weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(int(parameter.grad.count_nonzero()),0)
        self.model.zero_grad(set_to_none=True);self.model.eval();self.model.activation_checkpointing=False
    def test_strict_adapter_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs')as directory:
            path=Path(directory)/'model.pt';torch.save({'architecture':ARCH,'classes':list(CLASSES),'auxiliary_classes':list(AUX_CLASSES),'imgsz':640,'state_dict':self.model.state_dict()},path)
            adapter=ConvnextResearchPresence(path)
            with torch.no_grad():a=adapter.model(self.input);b=self.model(self.input)
            self.assertTrue(torch.equal(a,b))
    def test_adapter_rejects_wrong_public_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs')as directory:
            path=Path(directory)/'wrong.pt';torch.save({'architecture':ARCH,'classes':list(reversed(CLASSES)),'auxiliary_classes':list(AUX_CLASSES),'imgsz':640,'state_dict':{}},path)
            with self.assertRaises(ValueError):ConvnextResearchPresence(path)

if __name__=='__main__':unittest.main()
