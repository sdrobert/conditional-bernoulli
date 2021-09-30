r"""Implementations of batched estimators of z

The functions in this module can estimate values

.. math::

    z = E_{b \sim P_\theta}[I[\sum_t b_t = \ell_*] G_\theta(b|x)]

simultaneously for batches of :math:`\ell_*` and :math:`x` as well as its derivative
with respect to parameters the
"""
import config
import abc
from forward_backward import extract_relevant_odds_forward, log_R_forward
from typing import Callable, List, Tuple
import torch
import math
from utils import enumerate_bernoulli_support

Theta = List[torch.Tensor]
PSampler = IndependentLogOdds = Callable[[torch.Tensor, Theta], torch.Tensor]
LogPDensity = Callable[[torch.Tensor, torch.Tensor, Theta], torch.Tensor]
LogGDensity = IndependentLogLikelihoods = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Theta], torch.Tensor
]


class Estimator(metaclass=abc.ABCMeta):

    lp: LogPDensity
    lg: LogGDensity
    theta: Theta

    def __init__(self, lp: LogPDensity, lg: LogGDensity, theta: Theta) -> None:
        self.lp, self.lg, self.theta = lp, lg, theta

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
        b = b.t().to(x.dtype)  # (tmax, smax)
        x = x.repeat_interleave(lens, 1)
        y = y.repeat_interleave(lens, 0)
        assert x.size(1) == b.size(1)
        lp = self.lp(b, x, self.theta)
        lg = self.lg(y, b, x, self.theta)
        lp_lg = lp + lg
        lp_lg_max = lp_lg.max().detach()
        zhat = (lp_lg - lp_lg_max).exp().sum().log() + lp_lg_max - math.log(nmax)
        return zhat.detach(), zhat


class RejectionEstimator(Estimator):

    psampler: PSampler
    mc_samples: int

    def __init__(
        self,
        psampler: PSampler,
        num_mc_samples: int,
        lp: LogPDensity,
        g: LogGDensity,
        theta: Theta,
    ) -> None:
        self.psampler = psampler
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, g, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        total = nmax * self.num_mc_samples
        x = x.repeat_interleave(self.num_mc_samples, 1)
        y = y.repeat_interleave(self.num_mc_samples, 0)
        lmax = lmax.repeat_interleave(self.num_mc_samples, 0)
        b = self.psampler(x, self.theta).t()
        accept_mask = (b.sum(1) == lmax).unsqueeze(1)
        if not accept_mask.any():
            # no samples were accepted. We set zhat to -inf. To acquire a zero gradient,
            # we sum all the parameters in self.theta and mask 'em out
            v = self.theta[0].sum()
            for x in self.theta[1:]:
                v += x.sum()
            v = v.masked_fill(torch.tensor(True, device=v.device), 0.0)
            return torch.tensor(-float("inf"), device=v.device), v
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
        lg = self.lg(y, b, x, self.theta)
        lg_max = lg.max().detach()
        g = (lg - lg_max).exp()
        zhat_ = g.sum()
        back = zhat_ + (lp * g.detach()).sum()
        # the back / zhat_ is the derivative of the log without actually using the log
        # (which isn't defined for negative values). Since the numerator and denominator
        # would both be multiplied by 'exp(lg_max) / total', we cancel them for
        # numerical stability
        return (
            zhat_.detach().log() + lg_max - math.log(total),
            (back / zhat_.detach()).masked_fill(zhat_ == 0, 0.0),
        )


class ConditionalBernoulliEstimator(Estimator):

    iw: IndependentLogOdds
    ig: IndependentLogLikelihoods

    def __init__(
        self,
        ilw: IndependentLogOdds,
        ilg: IndependentLogLikelihoods,
        lp: LogPDensity,
        lg: LogGDensity,
        theta: Theta,
    ) -> None:
        self.ilw = ilw
        self.ilg = ilg
        super().__init__(lp, lg, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ilw = self.ilw(x, self.theta)  # (tmax, nmax)
        ilw_f = extract_relevant_odds_forward(
            ilw.t(), lmax, True, config.EPS_INF
        )  # (l, nmax, d)
        ilg = self.ilg(x, y, lmax, self.theta)  # (l, d, nmax)
        iwg_f = ilw_f + ilg.transpose(1, 2)  # (l, nmax, d)
        log_r = log_R_forward(iwg_f, lmax, batch_first=True)  # (nmax)
        log_denom = ilw.t().exp().log1p().sum(1)  # (nmax)
        zhat = (log_r - log_denom).logsumexp(0) - math.log(log_r.size(0))
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


def _lg(
    y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
) -> torch.Tensor:
    weights = theta[1]
    y_act = x.squeeze(2).t() * weights  # (nmax, tmax)
    return (-((y - y_act) ** 2) * b.t()).sum(1)


def _ilw(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return logits.unsqueeze(1).expand_as(x.squeeze(2))


def _ilg(
    x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor, theta: Theta
) -> torch.Tensor:
    weights = theta[1]
    y_act = x.squeeze(2).t() * weights  # (nmax, tmax)
    nse = -((y - y_act) ** 2)
    return extract_relevant_odds_forward(nse, lmax, True, config.EPS_INF).transpose(
        1, 2
    )


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
    zhat_exp = torch.zeros((1,))
    for n in range(nmax):
        x_n = x[:, n : n + 1]
        y_n = y[n : n + 1]
        b_n, _ = enumerate_bernoulli_support(tmax, lmax[n : n + 1])
        b_n = b_n.t().float()
        lp_n = _lp(b_n, x_n, theta)
        lg_n = _lg(y_n, b_n, x_n, theta)
        zhat_exp += (lp_n + lg_n).exp().sum()
    zhat_exp = (zhat_exp / nmax).log()
    grad_zhat_exp = torch.autograd.grad(zhat_exp, theta)
    zhat_act, back_act = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])
    theta[0].backward(-grad_zhat_act[0])
    theta[1].backward(-grad_zhat_act[1])
    optimizer.step()
    zhat_act_2, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    assert zhat_act_2 > zhat_act


def test_rejection_estimator():
    torch.manual_seed(2)
    tmax, nmax, mc = 6, 3, 200000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = RejectionEstimator(_psampler, mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
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
    zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = ConditionalBernoulliEstimator(_ilw, _ilg, _lp, _lg, theta)(
        x, y, lmax
    )
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])
