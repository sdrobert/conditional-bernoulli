r"""Implementations of batched estimators of z

The functions in this module can estimate values

.. math::

    z = E_{b \sim P_\theta}[I[\sum_t b_t = \ell_*] G_\theta(b|x)]

simultaneously for batches of :math:`\ell_*` and :math:`x` as well as its derivative
with respect to parameters the
"""
import abc
from forward_backward import R_forward, extract_relevant_odds_forward
from typing import Callable, List, Tuple
import torch
from utils import enumerate_bernoulli_support

Theta = List[torch.Tensor]
PSampler = IndependentOdds = Callable[[torch.Tensor, Theta], torch.Tensor]
LogPDensity = Callable[[torch.Tensor, torch.Tensor, Theta], torch.Tensor]
GDensity = IndependentLikelihoods = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Theta], torch.Tensor
]


class Estimator(metaclass=abc.ABCMeta):

    lp: LogPDensity
    g: GDensity
    theta: Theta

    def __init__(self, lp: LogPDensity, g: GDensity, theta: Theta) -> None:
        self.lp, self.g, self.theta = lp, g, theta

    @abc.abstractmethod
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class EnumerateEstimator(Estimator):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tmax, nmax = x.size(0), x.size(1)
        b, lens = enumerate_bernoulli_support(tmax, lmax)
        b = b.t()  # (tmax, smax)
        x = x.repeat_interleave(lens, 1)
        y = y.repeat_interleave(lens, 0)
        assert x.size(1) == b.size(1)
        p = self.lp(b, x, self.theta).exp()
        g = self.g(y, b, x, self.theta)
        zhat = (p * g).sum() / nmax
        return zhat.detach(), zhat


class RejectionEstimator(Estimator):

    psampler: PSampler
    mc_samples: int

    def __init__(
        self,
        psampler: PSampler,
        num_mc_samples: int,
        lp: LogPDensity,
        g: GDensity,
        theta: Theta,
    ) -> None:
        self.psampler = psampler
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, g, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        x = x.repeat_interleave(self.num_mc_samples, 1)
        y = y.repeat_interleave(self.num_mc_samples, 0)
        lmax = lmax.repeat_interleave(self.num_mc_samples, 0)
        total = nmax * self.num_mc_samples
        b = self.psampler(x, self.theta).t()
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
        lp = self.lp(b, x, self.theta)
        g = self.g(y, b, x, self.theta)
        zhat = g.sum() / total
        return zhat.detach(), zhat + (g.detach() * lp).sum() / total


class ConditionalBernoulliEstimator(Estimator):

    iw: IndependentOdds
    ig: IndependentLikelihoods

    def __init__(
        self,
        iw: IndependentOdds,
        ig: IndependentLikelihoods,
        lp: LogPDensity,
        g: GDensity,
        theta: Theta,
    ) -> None:
        self.iw = iw
        self.ig = ig
        super().__init__(lp, g, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        iw = self.iw(x, self.theta)  # (tmax, nmax)
        iw_f = extract_relevant_odds_forward(iw.t(), lmax, True)  # (l, nmax, d)
        ig = self.ig(x, y, lmax, self.theta)  # (l, d, nmax)
        iwg_f = iw_f * ig.transpose(1, 2)  # (l, nmax, d)
        r = R_forward(iwg_f, lmax, batch_first=True)  # (nmax)
        denom = (1 + iw.t()).prod(1)  # (nmax)
        zhat = (r / denom).mean()  # (1,)
        return zhat.detach(), zhat


# TESTS

# a dumb little objective to test our estimators.
# theta = (logits, weights)
# y_act = sum_t weights_t * x_t * b_t
# try to match some target y by maximizing exp(-||y - y_act||^2)
def _lp(b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return torch.distributions.Bernoulli(logits=logits).log_prob(b.t()).sum(1)


def _psampler(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return torch.distributions.Bernoulli(logits=logits).sample([x.size(1)]).t()


def _g(y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
    weights = theta[1]
    y_act = x.squeeze(2).t() * weights  # (nmax, tmax)
    return (-((y - y_act) ** 2) * b.t()).sum(1).exp()


def _iw(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return logits.exp().unsqueeze(1).expand_as(x.squeeze(2))


def _ig(
    x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor, theta: Theta
) -> torch.Tensor:
    weights = theta[1]
    y_act = x.squeeze(2).t() * weights  # (nmax, tmax)
    nse = (-((y - y_act) ** 2)).exp()
    return extract_relevant_odds_forward(nse, lmax, True).transpose(1, 2)


def test_enumerate_estimator():
    torch.manual_seed(1)
    tmax, nmax = 10, 5
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    optimizer = torch.optim.Adam(theta)
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp = torch.zeros(1)
    for n in range(nmax):
        x_n = x[:, n : n + 1]
        y_n = y[n : n + 1]
        b_n, _ = enumerate_bernoulli_support(tmax, lmax[n : n + 1])
        b_n = b_n.t()
        p_n = _lp(b_n, x_n, theta).exp()
        g_n = _g(y_n, b_n, x_n, theta)
        zhat_exp += (p_n * g_n).sum()
    zhat_exp /= nmax
    grad_zhat_exp = torch.autograd.grad(zhat_exp, theta)
    zhat_act, back_act = EnumerateEstimator(_lp, _g, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])
    theta[0].backward(-grad_zhat_act[0])
    theta[1].backward(-grad_zhat_act[1])
    optimizer.step()
    zhat_act_2, _ = EnumerateEstimator(_lp, _g, theta)(x, y, lmax)
    assert zhat_act_2 > zhat_act


def test_rejection_estimator():
    torch.manual_seed(2)
    tmax, nmax, mc = 6, 3, 100000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _g, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = RejectionEstimator(_psampler, mc, _lp, _g, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    # N.B. in general, back_exp != back_act. Only their gradients should be roughly
    # equal
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_conditional_bernoulli_estimator():
    torch.manual_seed(3)
    tmax, nmax = 8, 12
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _g, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = ConditionalBernoulliEstimator(_iw, _ig, _lp, _g, theta)(
        x, y, lmax
    )
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])
