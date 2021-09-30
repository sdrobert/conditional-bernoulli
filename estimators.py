from distributions import ConditionalBernoulli, SimpleSamplingWithoutReplacement
import config
import abc
from forward_backward import extract_relevant_odds_forward, log_R_forward
from typing import Callable, List, Optional, Tuple
import torch
import math
from utils import enumerate_bernoulli_support

Theta = List[torch.Tensor]
PSampler = IndependentLogits = Callable[[torch.Tensor, Theta], torch.Tensor]
LogDensity = Callable[[torch.Tensor, torch.Tensor, Theta], torch.Tensor]
LogLikelihood = IndependentLogLikelihoods = LogInclusionEstimates = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Theta], torch.Tensor
]


class Estimator(metaclass=abc.ABCMeta):

    lp: LogDensity
    lg: LogLikelihood
    theta: Theta

    def __init__(self, lp: LogDensity, lg: LogLikelihood, theta: Theta) -> None:
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
        lp: LogDensity,
        g: LogLikelihood,
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

    iw: IndependentLogits
    ig: IndependentLogLikelihoods

    def __init__(
        self,
        ilw: IndependentLogits,
        ilg: IndependentLogLikelihoods,
        lp: LogDensity,
        lg: LogLikelihood,
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


class StaticSsworImportanceSampler(Estimator):

    num_mc_samples: int

    def __init__(
        self, num_mc_samples: int, lp: LogDensity, g: LogLikelihood, theta: Theta,
    ) -> None:
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, g, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        total = nmax * self.num_mc_samples
        sswor = SimpleSamplingWithoutReplacement(lmax, tmax)
        b = sswor.sample([self.num_mc_samples])
        lqb = sswor.log_prob(b).flatten().detach()  # mc * nmax
        b = b.flatten(end_dim=1).t()  # tmax, mc * nmax
        x = x.unsqueeze(1).expand(tmax, self.num_mc_samples, nmax, fmax).flatten(1, 2)
        y = y.expand([self.num_mc_samples] + list(y.size())).flatten(end_dim=1)
        lpb = self.lp(b, x, self.theta)
        lg = self.lg(y, b, x, self.theta)
        zhat = (lg + lpb - lqb).logsumexp(0) - math.log(total)
        return zhat.detach(), zhat


class AisImhEstimator(Estimator, metaclass=abc.ABCMeta):

    num_mc_samples: int
    x: Optional[torch.Tensor]
    y: Optional[torch.Tensor]
    lmax: Optional[torch.Tensor]

    def __init__(
        self, num_mc_samples: int, lp: LogDensity, lg: LogLikelihood, theta: Theta
    ) -> None:
        self.num_mc_samples = num_mc_samples
        self.x = self.y = self.lmax = None
        super().__init__(lp, lg, theta)

    def clear(self) -> None:
        self.x = self.y = self.lmax = None

    def initialize_proposal(self) -> None:
        pass

    def get_first_sample(self) -> torch.Tensor:
        # use sswor until we have a sample in the support of pi
        assert self.x is not None and self.y is not None and self.lmax is not None
        tmax, nmax, _ = self.x.size()
        sswor = SimpleSamplingWithoutReplacement(self.lmax, tmax, tmax)
        b = torch.empty_like(self.x[..., 0])
        is_supported = torch.zeros((nmax,), device=self.x.device, dtype=torch.bool)
        while not is_supported.all():
            b_ = sswor.sample().t()  # (tmax, lmax)
            b_[:, is_supported] = b[:, is_supported]
            b = b_
            lp = self.lp(b, self.x, self.theta)
            lg = self.lg(self.y, b, self.x, self.theta)
            is_supported = ~(lp + lg).isinf()
        return b

    def update_proposal(self, b: torch.Tensor, mc_sample_num: int) -> None:
        pass

    @abc.abstractmethod
    def sample_proposal(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def proposal_log_prob(self, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.x, self.y, self.lmax = x, y, lmax
        nmax = x.size(1)
        total = nmax * self.num_mc_samples
        self.initialize_proposal()
        b_last = self.get_first_sample()
        zhat = torch.full_like(lmax, config.EPS_INF)
        for mc in range(1, self.num_mc_samples + 1):
            with torch.no_grad():
                lomega_last = (
                    self.lp(b_last, x, self.theta)
                    + self.lg(y, b_last, x, self.theta)
                    - self.proposal_log_prob(b_last)
                )
            b_prop = self.sample_proposal()
            lomega_prop = (
                self.lp(b_prop, x, self.theta)
                + self.lg(y, b_prop, x, self.theta)
                - self.proposal_log_prob(b_prop)
            )
            zhat = zhat.logaddexp(lomega_prop)
            lomega_prop = lomega_prop.detach().clamp_min(config.EPS_INF)

            with torch.no_grad():
                # N.B. This is intentionally the previous sample, not the current one.
                self.update_proposal(b_last, mc)

            choice = torch.bernoulli((lomega_prop - lomega_last).sigmoid_())
            not_choice = 1 - choice
            b_last = b_prop * choice + b_last * not_choice
        self.clear()
        zhat = zhat.logsumexp(0) - math.log(total)
        return zhat.detach(), zhat


class CbAisImhEstimator(AisImhEstimator):

    cb: Optional[ConditionalBernoulli]
    li: Optional[torch.Tensor]
    alpha: float
    li_requires_norm: bool
    lie: LogInclusionEstimates

    def __init__(
        self,
        lie: LogInclusionEstimates,
        num_mc_samples: int,
        lp: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
        alpha: float = 2.0,
        li_requires_norm: bool = True,
    ) -> None:
        self.lie, self.alpha, self.li_requires_norm = lie, alpha, li_requires_norm
        self.cb = self.li = None
        super().__init__(num_mc_samples, lp, lg, theta)

    def clear(self) -> None:
        self.cb = self.li = None
        return super().clear()

    def initialize_proposal(self) -> None:
        assert self.x is not None and self.lmax is not None
        self.cb = ConditionalBernoulli(
            self.lmax, logits=torch.randn_like(self.x[..., 0].t())
        )
        self.li = None

    def update_proposal(self, b: torch.Tensor, mc_sample_num: int) -> None:
        assert self.lmax is not None and self.y is not None and self.x is not None
        li_new = self.lie(self.y, b, self.x, self.theta)
        if mc_sample_num == 1:
            self.li = li_new
        else:
            assert self.li is not None
            self.li = (self.li + math.log(mc_sample_num - 1)).logaddexp(
                li_new
            ) - math.log(mc_sample_num)
        li = self.li
        if self.li_requires_norm:
            ldenom = (self.alpha * (li.max(0)[0].exp_() - 1).clamp_min_(0.0)).log1p_()
            li = li - ldenom
        assert (li <= 0.0).all()
        logits = li - torch.log1p(-li.exp()).clamp_min_(config.EPS_INF)
        self.cb = ConditionalBernoulli(self.lmax, logits=logits.t())

    def sample_proposal(self) -> torch.Tensor:
        assert self.cb is not None
        return self.cb.sample().t()

    def proposal_log_prob(self, b: torch.Tensor) -> torch.Tensor:
        assert self.cb is not None
        return self.cb.log_prob(b.t())


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


def _lie(
    y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
) -> torch.Tensor:
    logits, weights = theta
    y_act = x.squeeze(2).t() * weights  # (nmax, tmax)
    nse = -((y - y_act) ** 2)
    return (nse + logits).t()  # (tmax, nmax)


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


def test_sswor_is():
    torch.manual_seed(4)
    tmax, nmax, mc = 7, 2, 300000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = StaticSsworImportanceSampler(mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


class _DummyAisImhEstimator(AisImhEstimator):
    def initialize_proposal(self) -> None:
        tmax = self.x.size(0)
        self.sswor = SimpleSamplingWithoutReplacement(self.lmax, tmax, tmax)

    def sample_proposal(self) -> torch.Tensor:
        return self.sswor.sample().t()

    def proposal_log_prob(self, b: torch.Tensor) -> torch.Tensor:
        l = self.sswor.log_prob(b.t())
        return l


def test_dummy_ais():
    torch.manual_seed(5)
    tmax, nmax, mc = 5, 4, 50000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = _DummyAisImhEstimator(mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_cb_ais_imh():
    torch.manual_seed(6)
    tmax, nmax, mc = 5, 4, 5000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act = CbAisImhEstimator(_lie, mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)
