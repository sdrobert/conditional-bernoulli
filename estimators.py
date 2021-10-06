from distributions import ConditionalBernoulli, SimpleRandomSamplingWithoutReplacement
import config
import abc
from forward_backward import extract_relevant_odds_forward, log_R_forward
from typing import Callable, List, Optional, Tuple
import torch
import math
from utils import enumerate_bernoulli_support, enumerate_gibbs_partition

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class EnumerateEstimator(Estimator):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return zhat.detach(), zhat, torch.tensor(float("nan"))


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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return (
                torch.tensor(-float("inf"), device=v.device),
                v,
                torch.tensor(float("nan")),
            )
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
            torch.tensor(float("nan")),
        )


class ExtendedConditionalBernoulliEstimator(Estimator):

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ilw = self.ilw(x, self.theta)  # (tmax, nmax)
        ilw_f = extract_relevant_odds_forward(
            ilw.t(), lmax, True, config.EPS_INF
        )  # (l, nmax, d)
        ilg = self.ilg(x, y, lmax, self.theta)  # (l, d, nmax)
        iwg_f = ilw_f + ilg.transpose(1, 2)  # (l, nmax, d)
        log_r = log_R_forward(iwg_f, lmax, batch_first=True)  # (nmax)
        log_denom = ilw.t().exp().log1p().sum(1)  # (nmax)
        zhat = (log_r - log_denom).logsumexp(0) - math.log(log_r.size(0))
        return zhat.detach(), zhat, torch.tensor(float("nan"))


class StaticSrsworIsEstimator(Estimator):

    num_mc_samples: int

    def __init__(
        self, num_mc_samples: int, lp: LogDensity, g: LogLikelihood, theta: Theta,
    ) -> None:
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, g, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        total = nmax * self.num_mc_samples
        srswor = SimpleRandomSamplingWithoutReplacement(lmax, tmax)
        b = srswor.sample([self.num_mc_samples])
        lqb = srswor.log_prob(b).flatten().detach()  # mc * nmax
        b = b.flatten(end_dim=1).t()  # tmax, mc * nmax
        x = x.unsqueeze(1).expand(tmax, self.num_mc_samples, nmax, fmax).flatten(1, 2)
        y = y.expand([self.num_mc_samples] + list(y.size())).flatten(end_dim=1)
        lpb = self.lp(b, x, self.theta)
        lg = self.lg(y, b, x, self.theta)
        lomega = lg + lpb - lqb
        zhat = lomega.logsumexp(0) - math.log(total)
        with torch.no_grad():
            lomega = lomega.reshape(self.num_mc_samples, nmax)
            norm = lomega.logsumexp(0)
            lomega = lomega - norm
            log_ess = (
                2 * math.log(self.num_mc_samples) - (2 * lomega).logsumexp(0).mean()
            )
        return zhat.detach(), zhat, log_ess


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
        # use srswor until we have a sample in the support of pi
        assert self.x is not None and self.y is not None and self.lmax is not None
        tmax, nmax, _ = self.x.size()
        srswor = SimpleRandomSamplingWithoutReplacement(self.lmax, tmax, tmax)
        b = torch.empty_like(self.x[..., 0])
        is_supported = torch.zeros((nmax,), device=self.x.device, dtype=torch.bool)
        while not is_supported.all():
            b_ = srswor.sample().t()  # (tmax, lmax)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.x, self.y, self.lmax = x, y, lmax
        nmax = x.size(1)
        total = nmax * self.num_mc_samples
        self.initialize_proposal()
        b_last = self.get_first_sample()
        lomegas = []
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
            lomegas.append(lomega_prop)

            with torch.no_grad():
                # N.B. This is intentionally the previous sample, not the current one.
                self.update_proposal(b_last, mc)
                choice = torch.bernoulli(
                    (lomega_prop - lomega_last).clamp_min_(config.EPS_INF).sigmoid_()
                )

            not_choice = 1 - choice
            b_last = b_prop * choice + b_last * not_choice
        self.clear()
        lomegas = torch.stack(lomegas)
        zhat = lomegas.flatten().logsumexp(0) - math.log(total)
        with torch.no_grad():
            lomegas = lomegas.reshape(self.num_mc_samples, nmax)
            norm = lomegas.logsumexp(0)
            lomegas = lomegas - norm
            log_ess = (
                2 * math.log(self.num_mc_samples) - (2 * lomegas).logsumexp(0).mean()
            )
        return zhat.detach(), zhat, log_ess


class DummyAisImhEstimator(AisImhEstimator):
    def initialize_proposal(self) -> None:
        tmax = self.x.size(0)
        self.srswor = SimpleRandomSamplingWithoutReplacement(self.lmax, tmax, tmax)

    def sample_proposal(self) -> torch.Tensor:
        return self.srswor.sample().t()

    def proposal_log_prob(self, b: torch.Tensor) -> torch.Tensor:
        l = self.srswor.log_prob(b.t())
        return l


class CbAisImhEstimator(AisImhEstimator):

    cb: Optional[ConditionalBernoulli]
    li: Optional[torch.Tensor]
    lie: LogInclusionEstimates

    def __init__(
        self,
        lie: LogInclusionEstimates,
        num_mc_samples: int,
        lp: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
    ) -> None:
        self.lie = lie
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
        self.li = self.cb.probs.t().log()

    def update_proposal(self, b: torch.Tensor, mc_sample_num: int) -> None:
        assert self.lmax is not None and self.y is not None and self.x is not None
        li_new = self.lie(self.y, b, self.x, self.theta)
        self.li = (self.li + math.log(mc_sample_num)).logaddexp(li_new) - math.log(
            mc_sample_num + 1
        )
        li = self.li.clamp(config.EPS_INF, config.EPS_0)
        logits = li.t() - torch.log1p(-li.t().exp())
        self.cb = ConditionalBernoulli(self.lmax, logits=logits)

    def sample_proposal(self) -> torch.Tensor:
        assert self.cb is not None
        return self.cb.sample().t()

    def proposal_log_prob(self, b: torch.Tensor) -> torch.Tensor:
        assert self.cb is not None
        return self.cb.log_prob(b.t())


def gibbs_log_inclusion_estimates(
    y: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    lp: LogDensity,
    lg: LogLikelihood,
    theta: Theta,
    neg_inf: float = config.EPS_INF,
) -> torch.Tensor:

    device = b.device
    tmax, nmax = b.size()

    # b always contributes to pi(tau_ell = t|tau_{-ell})
    lyb = lp(b, x, theta) + lg(y, b, x, theta)  # (nmax,)

    # determine the other contributors
    bp, lens = enumerate_gibbs_partition(b)  # (tmax, nmax*), (nmax, lmax_max)
    lmax_max = lens.size(1)
    if not lmax_max:
        # there are no highs anywhere
        return torch.full((tmax, nmax), neg_inf, device=device, dtype=lyb.dtype)
    lens_ = lens.sum(1)
    y = y.repeat_interleave(lens_, 0)  # (nmax*, ...)
    x = x.repeat_interleave(lens_, 1)  # (tmax, nmax*, fmax)
    lybp = lp(bp, x, theta) + lg(y, bp, x, theta)  # (nmax*)
    b = b.long()
    bp = bp.long()

    # pad bp into (nmax, lmax_max, lens_max) tensor
    lens_max = int(lens.max().item())
    lens_mask = lens.unsqueeze(2) > torch.arange(lens_max, device=device)
    lybp = torch.full(
        (nmax, lmax_max, lens_max), -float("inf"), device=device
    ).masked_scatter(lens_mask, lybp)

    # determine normalization constant (b is part of all of them)
    norm = lybp.logsumexp(2).logaddexp(lyb.unsqueeze(1))  # (nmax, lmax_max)

    # determine the contribution to the inclusions for the samples b.
    # We add the mass to pi(b_t = 1) only when pi(tau_ell = t|tau_{-ell}), *not*
    # necessarily whenever t is some element of tau. To do this, we use the cumulative
    # sum on b to determine what number event a high is
    range_ = torch.arange(1, lmax_max + 1, device=device)
    cb = (b * b.cumsum(0)).unsqueeze(2) == range_  # (tmax, nmax, lmax_max)
    lyb = lyb.unsqueeze(1) - norm  # (nmax, lmax_max)
    lie_b = (cb * lyb + (~cb) * neg_inf).logsumexp(2)

    # now the contribution to the inclusions by bp.
    match = range_.view(1, lmax_max, 1).expand(nmax, lmax_max, lens_max)[lens_mask]
    cbp = bp * bp.cumsum(0) == match
    lybp = (lybp - norm.unsqueeze(2)).masked_select(lens_mask)  # (nmax*)
    lie_bp = cbp * lybp + (~cbp) * neg_inf  # (tmax, nmax*)
    lens__max = int(lens_.max().item())
    lens_mask = lens_.unsqueeze(1) > torch.arange(lens__max, device=device)
    lie_bp = (
        torch.full((tmax, nmax, lens__max), -float("inf"), device=device)
        .masked_scatter(lens_mask.unsqueeze(0), lie_bp)
        .logsumexp(2)
    )

    return torch.logaddexp(lie_b, lie_bp)


class GibbsCbAisImhEstimator(CbAisImhEstimator):
    def __init__(
        self, num_mc_samples: int, lp: LogDensity, lg: LogLikelihood, theta: Theta,
    ) -> None:
        lie = self._lie
        super().__init__(lie, num_mc_samples, lp, lg, theta)

    def _lie(self, y, b, x, theta):
        return gibbs_log_inclusion_estimates(y, b, x, self.lp, self.lg, theta)


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
    return (nse + logits - logits.exp().log1p()).t()  # (tmax, nmax)


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
    zhat_act, back_act, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])
    theta[0].backward(-grad_zhat_act[0])
    theta[1].backward(-grad_zhat_act[1])
    optimizer.step()
    zhat_act_2, _, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
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
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, _ = RejectionEstimator(_psampler, mc, _lp, _lg, theta)(
        x, y, lmax
    )
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_extended_conditional_bernoulli_estimator():
    torch.manual_seed(3)
    tmax, nmax = 8, 12
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, _ = ExtendedConditionalBernoulliEstimator(
        _ilw, _ilg, _lp, _lg, theta
    )(x, y, lmax)
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0])
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1])


def test_static_srswor_is_estimator():
    torch.manual_seed(4)
    tmax, nmax, mc = 7, 2, 300000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, log_ess = StaticSrsworIsEstimator(mc, _lp, _lg, theta)(
        x, y, lmax
    )
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert not log_ess.isnan()
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


# def test_dummy_ais():
#     torch.manual_seed(5)
#     tmax, nmax, mc = 5, 4, 50000
#     x = torch.randn((tmax, nmax, 1))
#     logits = torch.randn((tmax,), requires_grad=True)
#     weights = torch.randn((tmax,), requires_grad=True)
#     theta = [logits, weights]
#     y = torch.randn(nmax, tmax)
#     lmax = torch.randint(tmax + 1, size=(nmax,))
#     zhat_exp, back_exp = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
#     grad_zhat_exp = torch.autograd.grad(back_exp, theta)
#     zhat_act, back_act = _DummyAisImhEstimator(mc, _lp, _lg, theta)(x, y, lmax)
#     grad_zhat_act = torch.autograd.grad(back_act, theta)
#     assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
#     assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
#     assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_cb_ais_imh():
    torch.manual_seed(6)
    tmax, nmax, mc = 5, 4, 1024
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, log_ess = CbAisImhEstimator(_lie, mc, _lp, _lg, theta)(
        x, y, lmax
    )
    assert not log_ess.isnan()
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_gibbs_log_inclusion_estimates():
    torch.manual_seed(7)
    tmax, nmax = 12, 123
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,))
    weights = torch.randn((tmax,))
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    b = torch.bernoulli(logits.sigmoid().unsqueeze(1).expand(tmax, nmax))
    lie = gibbs_log_inclusion_estimates(y, b, x, _lp, _lg, theta)
    assert lie.size() == torch.Size([tmax, nmax])

    # P(tau_ell = t|tau_{-ell}) > 0 for at most two values of ell
    assert (lie <= math.log(2)).all()

    assert torch.allclose(lie.t().exp().sum(1), b.sum(0))


def test_gibbs_cb_ais_imh():
    torch.manual_seed(8)
    tmax, nmax, mc = 5, 4, 10000
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, log_ess = GibbsCbAisImhEstimator(mc, _lp, _lg, theta)(
        x, y, lmax
    )
    assert not log_ess.isnan()
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)
