r"""Implementations of batched estimators of z

The functions in this module can estimate values

.. math::

    z = E_{b \sim P_\theta}[I[\sum_t b_t = \ell_*] G_\theta(b|x)]

simultaneously for batches of :math:`\ell_*` and :math:`x` as well as its derivative
with respect to parameters the
"""
import abc
from typing import Callable, List, Optional, Tuple
import torch
from utils import enumerate_bernoulli_support

Theta = List[torch.Tensor]
PSampler = Callable[[torch.Tensor, Theta], torch.Tensor]
LogPDensity = Callable[[torch.Tensor, torch.Tensor, Theta], torch.Tensor]
GDensity = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Theta], torch.Tensor]
ZHatAndGrad = Tuple[torch.Tensor, Theta]


class Estimator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lmax: torch.Tensor,
        theta: Theta,
        lP: LogPDensity,
        G: GDensity,
    ) -> ZHatAndGrad:
        raise NotImplementedError()


class EnumerateEstimator(Estimator):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lmax: torch.Tensor,
        theta: Theta,
        lP: LogPDensity,
        G: GDensity,
    ) -> ZHatAndGrad:
        tmax, nmax = x.size(0), x.size(1)
        b, lens = enumerate_bernoulli_support(tmax, lmax)
        b = b.t()  # (tmax, smax)
        x = x.repeat_interleave(lens, 1)
        y = y.repeat_interleave(lens, 0)
        assert x.size(1) == b.size(1)
        p = lP(b, x, theta).exp()
        g = G(y, b, x, theta)
        zhat = (p * g).sum() / nmax
        nabla_zhat = list(torch.autograd.grad(zhat, theta))
        return zhat, nabla_zhat


class RejectionEstimator(Estimator):

    psampler: PSampler
    mc_samples: int

    def __init__(self, psampler: PSampler, num_mc_samples: int) -> None:
        super().__init__()
        self.psampler = psampler
        self.num_mc_samples = num_mc_samples

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lmax: torch.Tensor,
        theta: Theta,
        lP: LogPDensity,
        G: GDensity,
    ) -> ZHatAndGrad:
        tmax, nmax, fmax = x.size()
        x = x.repeat_interleave(self.num_mc_samples, 1)
        y = y.repeat_interleave(self.num_mc_samples, 0)
        lmax = lmax.repeat_interleave(self.num_mc_samples, 0)
        total = nmax * self.num_mc_samples
        b = self.psampler(x, theta).t()
        accept_mask = (b.sum(1) == lmax).unsqueeze(1)
        b = b.masked_select(accept_mask).view(-1, tmax).t()
        x = (
            x.transpose(0, 1)
            .masked_select(accept_mask.unsqueeze(2))
            .view(-1, tmax, fmax)
            .transpose(0, 1)
        )
        y_shape = list(y.shape)
        y_shape[0] = b.size(1)
        y = (
            y.view(nmax * self.num_mc_samples, -1)
            .masked_select(accept_mask)
            .view(y_shape)
        )
        lp = lP(b, x, theta)
        g = G(y, b, x, theta)
        zhat = g.sum() / total
        nabla_zhat_g = torch.autograd.grad(zhat, theta, allow_unused=True)
        nabla_zhat_lp = torch.autograd.grad(
            (g.detach() * lp).sum() / total, theta, allow_unused=True
        )
        nabla_zhat = []
        for a, b in zip(nabla_zhat_g, nabla_zhat_lp):
            if a is None:
                if b is None:
                    raise RuntimeError(
                        "One of the differentiated Tensors appears not to have been "
                        "used in the graph"
                    )
                nabla_zhat.append(b)
            elif b is None:
                nabla_zhat.append(a)
            else:
                nabla_zhat.append(a + b)
        return zhat, nabla_zhat


# TESTS

# a dumb little objective to test our estimators.
# theta = (logits, weights)
# y_act = sum of weights * x corresponding to chosen b
# get y_act close to y via MSE
def _lP(b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return torch.distributions.Bernoulli(logits=logits).log_prob(b.t()).sum(1)


def _PSampler(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return torch.distributions.Bernoulli(logits=logits).sample([x.size(1)]).t()


def _G(y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
    y_act = (theta[1].unsqueeze(1) * x.squeeze(2) * b).sum(0)
    return (y_act - y) ** 2


def test_enumerate_estimator():
    torch.manual_seed(1)
    tmax, nmax = 10, 5
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp = torch.zeros(1)
    for n in range(nmax):
        x_n = x[:, n : n + 1]
        y_n = y[n : n + 1]
        b_n, _ = enumerate_bernoulli_support(tmax, lmax[n : n + 1])
        b_n = b_n.t()
        p_n = _lP(b_n, x_n, theta).exp()
        g_n = _G(y_n, b_n, x_n, theta)
        zhat_exp += (p_n * g_n).sum() / nmax
    nabla_zhat_exp = torch.autograd.grad(zhat_exp, theta)
    zhat_act, nabla_zhat_act = EnumerateEstimator()(x, y, lmax, theta, _lP, _G)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(nabla_zhat_exp[0], nabla_zhat_act[0])
    assert torch.allclose(nabla_zhat_exp[1], nabla_zhat_act[1])


def test_rejection_estimator():
    torch.manual_seed(2)
    tmax, nmax, mc = 6, 3, 100000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, nabla_zhat_exp = EnumerateEstimator()(x, y, lmax, theta, _lP, _G)
    zhat_act, nabla_zhat_act = RejectionEstimator(_PSampler, mc)(
        x, y, lmax, theta, _lP, _G
    )
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(nabla_zhat_exp[0], nabla_zhat_act[0], atol=1e-2)
    assert torch.allclose(nabla_zhat_exp[1], nabla_zhat_act[1], atol=1e-2)
