import math
import unittest
import torch
from safelog_ai.insplad_lower_flow import AffineCoupling, ProjectedNormalFlow, InspladLowerFlowMain
from safelog_ai.insplad_suspension import InspladSuspensionMain


class FlowTests(unittest.TestCase):
    def test_nonidentity_inverse_and_jacobian(self):
        torch.manual_seed(13)
        block = AffineCoupling(4, 8).double()
        with torch.no_grad():
            block.conditioner[-1].weight.normal_(0, .07)
            block.conditioner[-1].bias.normal_(0, .1)
        x = torch.randn(4, dtype=torch.double, requires_grad=True)
        y, logdet = block(x[None])
        self.assertTrue(torch.allclose(block.inverse(y), x[None], atol=1e-12, rtol=1e-12))
        jac = torch.autograd.functional.jacobian(lambda v: block(v[None])[0][0], x)
        self.assertTrue(torch.allclose(torch.linalg.slogdet(jac).logabsdet, logdet[0], atol=1e-12))
        flow = ProjectedNormalFlow(8, 4, 8, 2).double()
        flow.blocks[0].load_state_dict(block.state_dict())
        z, ld = flow.transform(x[None])
        self.assertTrue(torch.allclose(flow.inverse(z), x[None], atol=1e-12))
        expected = .5*(z.square()+math.log(2*math.pi)).sum(1)-ld
        self.assertTrue(torch.equal(flow.nll_latent(x[None]), expected))

    def test_projection_fixed_after_fit_and_no_nonfit_dependency(self):
        torch.manual_seed(7)
        fit = torch.randn(21, 12)
        a, b = ProjectedNormalFlow(12, 4, 8, 2), ProjectedNormalFlow(12, 4, 8, 2)
        a.fit_projection(fit)
        b.fit_projection(fit)
        self.assertTrue(torch.equal(a.projection, b.projection))
        before = {n: v.clone() for n, v in a.named_buffers()}
        a(torch.randn(6, 12)*100).mean().backward()
        self.assertTrue(all(torch.equal(before[n], v) for n, v in a.named_buffers()))
        self.assertTrue(torch.allclose(a.latent(fit).std(0, correction=0), torch.ones(4), atol=1e-5))
        with self.assertRaises(ValueError):
            a.fit_projection(fit[:3])

    def test_preserved_method_contracts_and_trainable_scope(self):
        for name in ('forward_surface', 'forward_real_mask', 'forward_insplad_glass', 'forward_insplad_suspension'):
            self.assertIs(getattr(InspladLowerFlowMain, name), getattr(InspladSuspensionMain, name))
        flow = ProjectedNormalFlow()
        self.assertEqual(sum(p.numel() for p in flow.parameters()), 43968)
        self.assertTrue(all(n.startswith('blocks.') for n, _ in flow.named_parameters()))


if __name__ == '__main__':
    unittest.main()
