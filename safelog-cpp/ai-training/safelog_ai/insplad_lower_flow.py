"""Normal-only density in a fixed FIT PCA subspace of the existing encoder.

The density is defined on projected descriptors, not full images or machine health.
"""
import math
import torch
from torch import nn
from .insplad_suspension import InspladSuspensionMain
from .fine_feature_probe import forward_with_fine
from .gearbox_supervised import gearbox_descriptor


class AffineCoupling(nn.Module):
    def __init__(self, dimensions=32, hidden=64):
        super().__init__()
        if dimensions % 2:
            raise ValueError('An even latent dimension is required')
        self.conditioner = nn.Sequential(nn.Linear(dimensions//2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dimensions))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def parameters_for(self, a):
        scale, shift = self.conditioner(a).chunk(2, dim=1)
        return 1.5*scale.tanh(), shift

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        scale, shift = self.parameters_for(a)
        return torch.cat((a, b*scale.exp()+shift), dim=1), scale.sum(1)

    def inverse(self, y):
        a, b = y.chunk(2, dim=1)
        scale, shift = self.parameters_for(a)
        return torch.cat((a, (b-shift)*(-scale).exp()), dim=1)


class ProjectedNormalFlow(nn.Module):
    def __init__(self, descriptor_dim=3072, latent_dim=32, hidden=64, blocks=6):
        super().__init__()
        self.register_buffer('mean', torch.zeros(descriptor_dim))
        self.register_buffer('std', torch.ones(descriptor_dim))
        self.register_buffer('center', torch.zeros(descriptor_dim))
        self.register_buffer('projection', torch.zeros(descriptor_dim, latent_dim))
        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_std', torch.ones(latent_dim))
        self.register_buffer('score_mean', torch.zeros(()))
        self.register_buffer('score_std', torch.ones(()))
        self.blocks = nn.ModuleList([AffineCoupling(latent_dim, hidden) for _ in range(blocks)])

    @torch.no_grad()
    def fit_projection(self, fit):
        """Call with the reserved normal FIT descriptors only, never other phases."""
        x = fit.detach().cpu().double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or len(x) <= self.projection.shape[1]:
            raise ValueError('Enough FIT rows and the declared descriptor width are required')
        if not torch.isfinite(x).all():
            raise ValueError('Finite FIT descriptors required')
        mean, std = x.mean(0), x.std(0, correction=0).clamp_min(.001)
        standardized = ((x-mean)/std).clamp(-8, 8)
        center = standardized.mean(0)
        _, _, vh = torch.linalg.svd(standardized-center, full_matrices=False)
        basis = vh[:self.projection.shape[1]].T.contiguous()
        pivots = basis.abs().argmax(0)
        basis *= torch.where(basis[pivots, torch.arange(basis.shape[1])] < 0, -1., 1.)
        # Store FP32 buffers first; latent moments use precisely that stored transform.
        for name, value in (('mean', mean), ('std', std), ('center', center), ('projection', basis)):
            getattr(self, name).copy_(value.to(getattr(self, name)))
        z = self.project(fit)
        self.latent_mean.copy_(z.mean(0))
        self.latent_std.copy_(z.std(0, correction=0).clamp_min(.001))

    def project(self, descriptor):
        return (((descriptor-self.mean)/self.std).clamp(-8, 8)-self.center)@self.projection

    def latent(self, descriptor):
        return (self.project(descriptor)-self.latent_mean)/self.latent_std

    def transform(self, x):
        logdet = x.new_zeros(len(x))
        for block in self.blocks:
            x, delta = block(x)
            x = x.flip(1)  # Fixed permutation, absolute determinant one.
            logdet = logdet+delta
        return x, logdet

    def inverse(self, z):
        for block in reversed(self.blocks):
            z = block.inverse(z.flip(1))
        return z

    def nll_latent(self, x):
        z, logdet = self.transform(x)
        return .5*(z.square()+math.log(2*math.pi)).sum(1)-logdet

    def forward(self, descriptor):
        return self.nll_latent(self.latent(descriptor))

    def bounded_score(self, descriptor):
        return torch.sigmoid((self(descriptor)-self.score_mean)/self.score_std)


class InspladLowerFlowMain(InspladSuspensionMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.insplad_lower_flow = ProjectedNormalFlow()

    def forward_insplad_lower_descriptor(self, descriptor):
        return self.insplad_lower_flow.bounded_score(descriptor)

    def forward_insplad_lower(self, image):
        return self.forward_insplad_lower_descriptor(gearbox_descriptor(forward_with_fine(self.backbone, image)))
