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
        device = x.device
        b, lens = enumerate_bernoulli_support(tmax, lmax)
        b = b.t().to(x.dtype)  # (tmax, smax)
        x = x.repeat_interleave(lens, 1)
        y = y.repeat_interleave(lens, 0)
        assert x.size(1) == b.size(1)
        lp = self.lp(b, x, self.theta)
        lg = self.lg(y, b, x, self.theta)
        lp_lg = lp + lg
        lens_max = int(lens.max())
        len_mask = lens.unsqueeze(1) > torch.arange(lens_max, device=device)
        lp_lg = torch.full((nmax, lens_max), -float("inf")).masked_scatter(
            len_mask, lp + lg
        )
        zhat = lp_lg.logsumexp(1).mean()
        return zhat.detach(), zhat, torch.tensor(float("nan"))


class RejectionEstimator(Estimator):

    psampler: PSampler
    mc_samples: int

    def __init__(
        self,
        psampler: PSampler,
        num_mc_samples: int,
        lp: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
    ) -> None:
        self.psampler = psampler
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, lg, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        device = x.device
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
            v = v.masked_fill(torch.tensor(True, device=device), 0.0)
            return (
                torch.tensor(-float("inf"), device=device),
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
        accept_mask = accept_mask.flatten()
        lp = self.lp(b, x, self.theta)
        lp = (
            torch.full((nmax * self.num_mc_samples,), config.EPS_INF, device=device)
            .masked_scatter(accept_mask, lp)
            .view(nmax, self.num_mc_samples)
        )
        lg = self.lg(y, b, x, self.theta)
        lg = (
            torch.full((nmax * self.num_mc_samples,), config.EPS_INF, device=device)
            .masked_scatter(accept_mask, lg)
            .view(nmax, self.num_mc_samples)
        )
        lg_max = lg.detach().max(1)[0]
        g = (lg - lg_max.unsqueeze(1)).exp()
        zhat_ = g.sum(1)  # (nmax,)
        back = zhat_ + (lp.clamp_min(config.EPS_INF) * g.detach()).sum(1)
        # the back / zhat_ is the derivative of the log without actually using the log
        # (which isn't defined for negative values). Since the numerator and denominator
        # would both be multiplied by 'exp(lg_max)/self.num_mc_samples', we cancel them
        # out for numerical stability
        return (
            (zhat_.detach().log() + lg_max).mean() - math.log(self.num_mc_samples),
            (back / zhat_.detach()).masked_fill(zhat_ == 0, 0.0).mean(),
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
        zhat = (log_r - log_denom).mean()
        return zhat.detach(), zhat, torch.tensor(float("nan"))


class StaticSrsworIsEstimator(Estimator):

    num_mc_samples: int

    def __init__(
        self, num_mc_samples: int, lp: LogDensity, lg: LogLikelihood, theta: Theta,
    ) -> None:
        self.num_mc_samples = num_mc_samples
        super().__init__(lp, lg, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        srswor = SimpleRandomSamplingWithoutReplacement(lmax, tmax)
        b = srswor.sample([self.num_mc_samples])
        lqb = srswor.log_prob(b).flatten().detach()  # mc * nmax
        b = b.flatten(end_dim=1).t()  # tmax, mc * nmax
        x = x.unsqueeze(1).expand(tmax, self.num_mc_samples, nmax, fmax).flatten(1, 2)
        y = y.expand([self.num_mc_samples] + list(y.size())).flatten(end_dim=1)
        lpb = self.lp(b, x, self.theta)
        lg = self.lg(y, b, x, self.theta)
        lomega = lg + lpb - lqb
        lomega = lomega.reshape(self.num_mc_samples, nmax)
        norm = lomega.logsumexp(0)
        zhat = norm.mean() - math.log(self.num_mc_samples)
        with torch.no_grad():
            lomega = lomega - norm
            log_ess = -(2 * lomega).logsumexp(0).mean()
        return zhat.detach(), zhat, log_ess


class ForcedSuffixIsEstimator(Estimator):

    psampler: PSampler
    num_mc_samples: int

    def __init__(
        self,
        psampler: PSampler,
        num_mc_samples: int,
        lp_full: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
    ) -> None:
        self.psampler = psampler
        self.num_mc_samples = num_mc_samples
        super().__init__(lp_full, lg, theta)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmax, nmax, fmax = x.size()
        device = x.device
        x = x.repeat_interleave(self.num_mc_samples, 1)
        y = y.repeat_interleave(self.num_mc_samples, 0)
        lmax = lmax.repeat_interleave(self.num_mc_samples, 0)
        b = self.psampler(x, self.theta).t()

        # force suffix to whatever value makes the correct number of highs
        cb = b.cumsum(1)
        remove_mask = lmax.unsqueeze(1) > cb - b
        add_mask = lmax.unsqueeze(1) - cb + b > torch.arange(
            tmax - 1, -1, -1, device=device
        )
        b = b * remove_mask + add_mask * (1 - b)
        assert torch.equal(b.long().sum(1), lmax)
        # N.B. the last element is always forced
        assert (~remove_mask[:, -1] | add_mask[:, -1]).all()

        lpb_full = self.lp(b.t(), x, self.theta)  # (tmax, nmax * mc)
        lp = lpb_full.sum(0)
        lg = self.lg(y, b.t(), x, self.theta)
        lqb = lpb_full.detach().t().masked_fill((~remove_mask) | add_mask, 0.0).sum(1)

        lomega = (lp + lg - lqb).view(nmax, self.num_mc_samples)
        norm = lomega.logsumexp(1)
        zhat = norm.mean() - math.log(self.num_mc_samples)
        with torch.no_grad():
            lomega = lomega - norm.unsqueeze(1)
            log_ess = -(2 * lomega).logsumexp(1).mean()
        return zhat.detach(), zhat, log_ess


class AisImhEstimator(Estimator, metaclass=abc.ABCMeta):

    num_mc_samples: int
    burn_in: int
    x: Optional[torch.Tensor]
    y: Optional[torch.Tensor]
    lmax: Optional[torch.Tensor]
    srswor: Optional[SimpleRandomSamplingWithoutReplacement]

    def __init__(
        self,
        num_mc_samples: int,
        lp: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
        burn_in: int = 0,
    ) -> None:
        self.num_mc_samples, self.burn_in = num_mc_samples, burn_in
        self.x = self.y = self.lmax = self.srswor = None
        super().__init__(lp, lg, theta)

    def clear(self) -> None:
        self.x = self.y = self.lmax = self.srswor = None

    def initialize_proposal(self) -> None:
        assert self.x is not None and self.y is not None and self.lmax is not None
        tmax = self.x.size(0)
        self.srswor = SimpleRandomSamplingWithoutReplacement(self.lmax, tmax, tmax)

    @torch.no_grad()
    def get_first_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # use srswor until we have a sample in the support of pi
        assert self.x is not None and self.y is not None and self.lmax is not None
        nmax = self.x.size(1)
        b = torch.empty_like(self.x[..., 0])
        is_supported = torch.zeros((nmax,), device=self.x.device, dtype=torch.bool)
        while not is_supported.all():
            b_ = self.srswor.sample().t()  # (tmax, lmax)
            b_[:, is_supported] = b[:, is_supported]
            b = b_
            lp = self.lp(b, self.x, self.theta)
            lg = self.lg(self.y, b, self.x, self.theta)
            lp_lg = lp + lg
            is_supported = ~(lp + lg).isinf()
        return b, lp_lg

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
        self.initialize_proposal()
        b_last, lp_lg = self.get_first_sample()
        srswor_lprob = self.srswor.log_prob(b_last.t())  # constant
        lomega_prop = lp_lg - srswor_lprob
        lomegas = []
        # nmax = self.x.size(1)
        # choices = torch.zeros((nmax,))
        for mc in range(1, self.num_mc_samples + 1):

            if mc > self.burn_in:
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
            else:
                lomega_last = lomega_prop.detach()  # last iteration's
                b_prop = self.srswor.sample().t()
                lomega_prop = (
                    self.lp(b_prop, x, self.theta)
                    + self.lg(y, b_prop, x, self.theta)
                    - srswor_lprob
                )

            lomegas.append(lomega_prop)

            with torch.no_grad():
                # N.B. This is intentionally the previous sample, not the current one.
                self.update_proposal(b_last, mc)
                choice = torch.bernoulli((lomega_prop - lomega_last).exp().clamp_max(1))
                # if mc > self.burn_in:
                #     choices += choice
                not_choice = 1 - choice
                b_last = b_prop * choice + b_last * not_choice
        # print("ar", choices.mean() / (self.num_mc_samples - self.burn_in))
        self.clear()
        lomegas = torch.stack(lomegas)
        norm = lomegas.logsumexp(0)
        zhat = norm.mean() - math.log(self.num_mc_samples)
        with torch.no_grad():
            lomegas = lomegas - norm
            log_ess = -(2 * lomegas).logsumexp(0).mean()
        return zhat.detach(), zhat, log_ess


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
        burn_in: int = 0,
    ) -> None:
        self.lie = lie
        self.cb = self.li = None
        super().__init__(num_mc_samples, lp, lg, theta, burn_in)

    def clear(self) -> None:
        self.cb = self.li = None
        return super().clear()

    def initialize_proposal(self) -> None:
        super().initialize_proposal()
        self.li = (
            (self.lmax.to(self.x.dtype).log() - math.log(self.x.size(0)))
            .expand_as(self.x[..., 0])
            .clamp(config.EPS_INF, config.EPS_0)
        )
        # doesn't matter what you put in. What's important is that they're all equal
        self.cb = ConditionalBernoulli(self.lmax, logits=self.li.t())

    def update_proposal(self, b: torch.Tensor, mc_sample_num: int) -> None:
        assert self.lmax is not None and self.y is not None and self.x is not None
        av_coeff = torch.as_tensor(mc_sample_num, device=b.device, dtype=b.dtype,)
        li_new = self.lie(self.y, b, self.x, self.theta)
        self.li = (self.li + av_coeff.log()).logaddexp(li_new) - (av_coeff + 1).log()
        li = self.li.t().clamp(config.EPS_INF, config.EPS_0)
        # if mc_sample_num % 10 == 1:
        #     print(li[0].exp()[:10])
        self.cb = ConditionalBernoulli(self.lmax, logits=li - (-li.exp()).log1p())

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
        self,
        num_mc_samples: int,
        lp: LogDensity,
        lg: LogLikelihood,
        theta: Theta,
        burn_in: int = 0,
    ) -> None:
        lie = self._lie
        super().__init__(lie, num_mc_samples, lp, lg, theta, burn_in)

    def _lie(self, y, b, x, theta):
        return gibbs_log_inclusion_estimates(y, b, x, self.lp, self.lg, theta)


# TESTS

# a dumb little objective to test our estimators.
# theta = (logits, weights)
# y_act = sum_t weights_t * x_t * b_t
# try to match some target y by maximizing exp(-||y - y_act||^2)
def _lp(
    b: torch.Tensor, x: torch.Tensor, theta: Theta, full: bool = False
) -> torch.Tensor:
    lp = torch.distributions.Bernoulli(logits=theta[0]).log_prob(b.t())
    if full:
        return lp.t()
    else:
        return lp.sum(1)


def _psampler(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    return torch.distributions.Bernoulli(logits=theta[0]).sample([x.size(1)]).t()


def _lg(
    y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
) -> torch.Tensor:
    y_act = x.squeeze(2).t() * theta[1]  # (nmax, tmax)
    return (-((y - y_act) ** 2) * b.t()).sum(1)


def _ilw(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    return theta[0].unsqueeze(1).expand_as(x.squeeze(2))


def _ilg(
    x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor, theta: Theta
) -> torch.Tensor:
    y_act = x.squeeze(2).t() * theta[1]  # (nmax, tmax)
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
        zhat_exp += (lp_n + lg_n).flatten().logsumexp(0)
    zhat_exp = zhat_exp / nmax
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


# class _DummyAisImhEstimator(AisImhEstimator):

#     sample_proposal = AisImhEstimator._sample_srswor
#     proposal_log_prob = AisImhEstimator._srswor_log_prob


# def test_dummy_ais():
#     torch.manual_seed(5)
#     tmax, nmax, mc = 5, 4, 50000
#     x = torch.randn((tmax, nmax, 1))
#     logits = torch.randn((tmax,), requires_grad=True)
#     weights = torch.randn((tmax,), requires_grad=True)
#     theta = [logits, weights]
#     y = torch.randn(nmax, tmax)
#     lmax = torch.randint(tmax + 1, size=(nmax,))
#     zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
#     grad_zhat_exp = torch.autograd.grad(back_exp, theta)
#     zhat_act, back_act, log_ess = _DummyAisImhEstimator(mc, _lp, _lg, theta)(x, y, lmax)
#     assert not log_ess.isnan()
#     grad_zhat_act = torch.autograd.grad(back_act, theta)
#     assert torch.allclose(zhat_exp, zhat_act, atol=1e-2)
#     assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-2)
#     assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-2)


def test_cb_ais_imh():
    torch.manual_seed(6)
    tmax, nmax, mc = 5, 4, 2048
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


def test_forced_suffix_is_estimator():
    torch.manual_seed(9)
    tmax, nmax, mc = 10, 100, 2 ** 10
    x = torch.randn((tmax, nmax, 1))
    logits = torch.randn((tmax,), requires_grad=True)
    weights = torch.randn((tmax,), requires_grad=True)
    theta = [logits, weights]
    y = torch.randn(nmax, tmax)
    lmax = torch.randint(tmax + 1, size=(nmax,))
    zhat_exp, back_exp, _ = EnumerateEstimator(_lp, _lg, theta)(x, y, lmax)
    grad_zhat_exp = torch.autograd.grad(back_exp, theta)
    zhat_act, back_act, log_ess = ForcedSuffixIsEstimator(
        _psampler, mc, lambda *z: _lp(*z, full=True), _lg, theta
    )(x, y, lmax)
    assert not log_ess.isnan()
    grad_zhat_act = torch.autograd.grad(back_act, theta)
    assert torch.allclose(zhat_exp, zhat_act, atol=1e-1)
    assert torch.allclose(grad_zhat_exp[0], grad_zhat_act[0], atol=1e-1)
    assert torch.allclose(grad_zhat_exp[1], grad_zhat_act[1], atol=1e-1)
