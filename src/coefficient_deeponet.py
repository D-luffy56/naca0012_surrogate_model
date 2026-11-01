"""Parametric DeepONet queried on discrete POD modes, not spatial coordinates.

True coefficients are training labels only. The trained trunk can be evaluated
once on all one-hot mode queries. This factorizes a multi-output regressor; it
does not imply generalization to unseen geometries, modes or input functions.
"""
import numpy as np
import torch
from torch import nn


def dense(sizes, activation='tanh'):
    act = {'tanh': nn.Tanh, 'relu': nn.ReLU}[activation]
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(act())
    return nn.Sequential(*layers)


class CoefficientDeepONet(nn.Module):
    def __init__(self, rank, hidden_units=(32, 32), activation='tanh', latent=32):
        super().__init__()
        if rank < 1 or latent < 1:
            raise ValueError('rank and latent must be positive')
        self.rank, self.latent = rank, latent
        self.branch = dense((2, *hidden_units, latent), activation)
        self.trunk = dense((rank, *hidden_units, latent), activation)
        self.bias = nn.Parameter(torch.zeros(rank))
        self.register_buffer('queries', torch.eye(rank))

    def forward(self, x):
        if x.shape[-1] != 2:
            raise ValueError('Expected standardized [Re, AoA] only')
        return self.branch(x) @ self.trunk(self.queries).T / self.latent**0.5 + self.bias


def weight_penalty(net):
    return sum(layer.weight.square().sum() for layer in net.modules()
               if isinstance(layer, nn.Linear))


def fit_coefficients(x, y, hidden_units=(32, 32), activation='tanh', latent=32,
                     seed=42, adam_steps=1500, lbfgs_steps=500, alpha=1e-4):
    """Training-only deterministic optimization. No validation-dependent stopping."""
    torch.manual_seed(seed)
    net = CoefficientDeepONet(y.shape[1], hidden_units, activation, latent)
    inputs = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    history = []
    def objective():
        error = (net(inputs)-targets).square().mean()
        return error + alpha/(2*len(inputs))*weight_penalty(net), error
    for step in range(adam_steps):
        optimizer.zero_grad()
        loss, error = objective()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == adam_steps-1:
            history.append({'stage':'adam','step':step+1,'objective':float(loss.detach()),
                            'coefficient_mse':float(error.detach())})
    optimizer = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=lbfgs_steps,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-7,
                                tolerance_change=1e-10, history_size=50)
    def closure():
        optimizer.zero_grad()
        loss, _ = objective()
        loss.backward()
        return loss
    if lbfgs_steps:
        optimizer.step(closure)
    with torch.no_grad():
        loss, error = objective()
    if not torch.isfinite(loss):
        raise FloatingPointError('Coefficient fit diverged')
    history.append({'stage':'lbfgs','step':lbfgs_steps,'objective':float(loss),
                    'coefficient_mse':float(error)})
    return net, history
