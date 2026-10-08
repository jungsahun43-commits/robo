"""Exact public transfer, single-pass geometry and useful dense gradients."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as ORIGINAL_ARCH
from safelog_ai.dense_auxiliary_classifier import (
    ARCH, DenseAuxiliaryClassifier, load_original_state, model_inventory,
    original_state, zero_dense_projection,
)
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm


CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]
SPLIT_SHA = "a" * 64


class DenseAuxiliaryModelTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(31)
        cls.base = AuxiliaryClassifier(7, pretrained=False)
        cls.checkpoint = {
            "architecture": ORIGINAL_ARCH,
            "classes": CLASSES,
            "auxiliary_classes": list(AUX_CLASSES),
            "imgsz": 640,
            "split_sha256": SPLIT_SHA,
            "state_dict": {name: value.detach().clone() for name, value in cls.base.state_dict().items()},
        }

    def setUp(self):
        self.base.load_state_dict(self.checkpoint["state_dict"], strict=True)
        self.model = DenseAuxiliaryClassifier(7, pretrained=False)
        self.proof = load_original_state(self.model, self.checkpoint, CLASSES, SPLIT_SHA)
        self.base.eval()
        self.model.eval()

    def test_original_324_tensors_exact_and_all_existing_outputs_unchanged(self):
        image = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            expected = self.base.forward_training(image)
            inherited = self.model.forward_training(image)
            dense_outputs = self.model.forward_dense_training(image)
            public = self.model(image)
            details = self.model.forward_details(image)
        self.assertEqual(len(inherited), 3)
        self.assertEqual(len(dense_outputs), 4)
        for index in range(3):
            self.assertTrue(torch.equal(expected[index], inherited[index]))
            self.assertTrue(torch.equal(expected[index], dense_outputs[index]))
        self.assertTrue(torch.equal(expected[0], public))
        self.assertTrue(torch.equal(expected[0], details[0]))
        self.assertTrue(torch.equal(expected[1], details[1]))
        self.assertTrue(all(torch.equal(value, self.checkpoint["state_dict"][name])
                            for name, value in original_state(self.model).items()))
        self.assertTrue(self.proof["shared_state_tensors_equal"])
        self.assertEqual(self.proof["shared_state_tensor_count"], 324)
        self.assertEqual(tuple(dense_outputs[3].shape), (2, 19, 16, 16))
        self.assertEqual(torch.count_nonzero(dense_outputs[3]).item(), 0)

    def test_representative_640_geometry_uses_one_backbone_pass_and_old_methods_skip_dense(self):
        calls = {"backbone": 0, "dense": 0}
        def counted(name):
            def hook(*unused): calls[name] += 1
            return hook
        backbone_hook = self.model.backbone.register_forward_pre_hook(counted("backbone"))
        dense_hook = self.model.dense_auxiliary_head.register_forward_pre_hook(counted("dense"))
        try:
            image = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                photo, maps, auxiliary, dense = self.model.forward_dense_training(image)
            self.assertEqual(calls, {"backbone": 1, "dense": 1})
            self.assertEqual(tuple(photo.shape), (1, 7))
            self.assertEqual(tuple(maps.shape), (1, 7, 80, 80))
            self.assertEqual(tuple(auxiliary.shape), (1, 19))
            self.assertEqual(tuple(dense.shape), (1, 19, 80, 80))
            self.assertTrue(all(torch.isfinite(value).all() for value in (photo, maps, auxiliary, dense)))
            with torch.no_grad():
                self.model.forward_training(image)
                self.model(image)
            self.assertEqual(calls, {"backbone": 3, "dense": 1})
        finally:
            backbone_hook.remove()
            dense_hook.remove()

    def test_real_first_projection_update_and_second_adapter_backbone_gradients(self):
        self.model.train()
        apply_frozen_batchnorm(self.model)
        before = {name: value.detach().clone() for name, value in original_state(self.model).items()}
        head = self.model.dense_auxiliary_head
        initial_adapter = {name: value.detach().clone() for name, value in head.high_adapter.named_parameters()}
        optimizer = torch.optim.SGD(head.parameters(), lr=.05)
        image = torch.randn(2, 3, 128, 128)
        for step in range(2):
            self.model.zero_grad(set_to_none=True)
            dense = self.model.forward_dense_training(image)[3]
            target = torch.zeros_like(dense)
            target[:, 0] = 1
            nn.functional.binary_cross_entropy_with_logits(dense, target).backward()
            for parameter in head.projection.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue((parameter.grad != 0).any())
            adapter_grad = head.high_adapter[0].weight.grad
            self.assertIsNotNone(adapter_grad)
            if step == 0:
                self.assertEqual(torch.count_nonzero(adapter_grad).item(), 0)
            else:
                self.assertTrue(torch.isfinite(adapter_grad).all())
                self.assertTrue((adapter_grad != 0).any())
                self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    and bool((parameter.grad != 0).any())
                                    for parameter in self.model.backbone.parameters()))
            optimizer.step()
        self.assertFalse(zero_dense_projection(self.model))
        # A random MobileNet with fixed BN can produce very small high features;
        # its adapter weight update may round to the same float32 values. The
        # gradient must still be nonzero, and an actual adapter parameter must
        # change on the second update.
        self.assertTrue(any(not torch.equal(initial_adapter[name], value)
                            for name, value in head.high_adapter.named_parameters()))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in original_state(self.model).items()))

    def test_seeded_adapter_does_not_consume_extra_global_rng_or_seed_cuda(self):
        torch.manual_seed(93)
        self.base.__class__(7, pretrained=False)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(93)
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CPU adapter must not reseed CUDA")):
            other = DenseAuxiliaryClassifier(7, pretrained=False, dense_seed=56)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertTrue(all(torch.equal(value, self.model.dense_auxiliary_head.state_dict()[name])
                            for name, value in other.dense_auxiliary_head.state_dict().items()))
        changed = DenseAuxiliaryClassifier(7, pretrained=False, dense_seed=57)
        self.assertFalse(torch.equal(other.dense_auxiliary_head.high_adapter[0].weight,
                                     changed.dense_auxiliary_head.high_adapter[0].weight))
        self.assertTrue(zero_dense_projection(changed))

    def test_inventory_has_four_new_tensors_no_new_bn_and_seven_public_classes(self):
        inventory = model_inventory(self.model)
        self.assertEqual(inventory["architecture"], ARCH)
        self.assertEqual(inventory["state_tensor_count"], 328)
        self.assertEqual(inventory["original_state_tensor_count"], 324)
        self.assertEqual(inventory["new_dense_head_state_tensor_count"], 4)
        self.assertEqual(inventory["new_dense_head_parameter_count"], 63499)
        self.assertEqual(inventory["parameter_count"], 3307650)
        self.assertEqual(inventory["dense_output_projection_parameter_count"], 1995)
        self.assertEqual(inventory["new_batchnorm_modules"], 0)
        self.assertEqual(inventory["public_class_count"], 7)
        self.assertEqual(inventory["dense_classes"], list(AUX_CLASSES))
        self.assertTrue(inventory["dense_output_projection_zero"])
        self.assertTrue(inventory["dense_head_trainable"])
        self.assertFalse(inventory["public_output_contract_changed"])
        self.assertFalse(inventory["dense_branch_directly_changes_public_outputs"])
        self.assertEqual(inventory["additional_backbone_passes"], 0)

    def test_strict_initializer_rejects_wrong_inventory_metadata_and_nonfinite_before_transfer(self):
        initial = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        for kind in ("missing", "unknown", "dtype", "nonfinite", "architecture", "split", "imgsz", "aux_order"):
            checkpoint = dict(self.checkpoint)
            checkpoint["state_dict"] = dict(self.checkpoint["state_dict"])
            if kind == "missing": checkpoint["state_dict"].pop("mix")
            elif kind == "unknown": checkpoint["state_dict"]["unexpected"] = torch.zeros(1)
            elif kind == "dtype": checkpoint["state_dict"]["mix"] = torch.zeros(7, dtype=torch.float64)
            elif kind == "nonfinite": checkpoint["state_dict"]["mix"] = torch.full((7,), float("nan"))
            elif kind == "architecture": checkpoint["architecture"] = ARCH
            elif kind == "split": checkpoint["split_sha256"] = "b" * 64
            elif kind == "imgsz": checkpoint["imgsz"] = True
            elif kind == "aux_order": checkpoint["auxiliary_classes"] = list(reversed(AUX_CLASSES))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_original_state(self.model, checkpoint, CLASSES, SPLIT_SHA)
            self.assertTrue(all(torch.equal(value, initial[name]) for name, value in self.model.state_dict().items()))
        with torch.no_grad(): self.model.dense_auxiliary_head.projection.bias[0] = 1
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            load_original_state(self.model, self.checkpoint, CLASSES, SPLIT_SHA)

    def test_offline_strict_dense_reload_and_invalid_construction(self):
        state = deepcopy(self.model.state_dict())
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("No network")):
            reloaded = DenseAuxiliaryClassifier(7, pretrained=False)
            reloaded.load_state_dict(state, strict=True)
            reloaded.eval()
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected = self.model.forward_dense_training(image)
            actual = reloaded.forward_dense_training(image)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))
        for kwargs in ({"classes": 19}, {"classes": True}, {"pretrained": True},
                       {"dense_seed": -1}, {"dense_seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DenseAuxiliaryClassifier(**kwargs)
