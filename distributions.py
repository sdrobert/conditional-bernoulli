"""Distributions, in particular the Conditional Bernoulli"""

from forward_backward import R_forward, extract_relevant_odds_forward
from typing import Optional, Union
import torch

import torch.distributions.constraints as constraints
from torch.distributions.utils import lazy_property


def poisson_binomial(w: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(w / (1 + w)).sum(-1)


class PoissonBinomial(torch.distributions.Distribution):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
        "odds": constraints.greater_than_eq(0.0),
    }

    def __init__(
        self,
        total_count: Union[torch.Tensor, int, None] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        odds: Optional[torch.Tensor] = None,
        validate_args=None,
    ):
        non_none = {x for x in (probs, logits, odds) if x is not None}
        if len(non_none) != 1:
            raise ValueError("Exactly one of probs, logits, or must be non-None.")
        param = non_none.pop()
        if param.dim() < 1:
            raise ValueError("parameter must be at least 1-dimensional")
        param_size = param.size()
        batch_size = param_size[:-1]
        tmax = param_size[-1]
        if total_count is None:
            mask = None
        else:
            self.total_count = (
                torch.as_tensor(total_count).expand(batch_size).type_as(param)
            )
            mask = self.total_count.unsqueeze(-1) <= torch.arange(tmax)
        if probs is not None:
            self.probs = self._param = (
                probs if mask is None else probs.masked_fill(mask, 0.0)
            )
        elif logits is not None:
            self.logits = self._param = (
                logits if mask is None else logits.masked_fill(mask, -float("inf"))
            )
        else:
            self.odds = self._param = (
                odds if mask is None else odds.masked_fill(mask, 0.0)
            )
        super(PoissonBinomial, self).__init__(batch_size, validate_args=validate_args)

    @lazy_property
    def total_count(self):
        return torch.full(self.batch_shape, self.odds.size(-1)).type_as(self._param)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    # probs <= odds <= logits <= probs

    @lazy_property
    def probs(self):
        return self.odds / (1 + self.odds)

    @lazy_property
    def odds(self):
        return self.logits.exp()

    @lazy_property
    def logits(self):
        return self.probs.logit()

    @lazy_property
    def log_r(self):
        odds, total_count = self.odds, self.total_count
        batch_shape = self.batch_shape
        batch_dims = len(batch_shape)
        if not batch_dims:
            odds, total_count = odds.unqueeze(0), total_count.view(1)
        else:
            odds, total_count = (
                odds.flatten(end_dim=batch_dims - 1),
                total_count.flatten(),
            )
        # unfortunately, the sums can get pretty big. This leads to NaNs in the
        # backward pass. We go straight into log here
        log_odds = odds.log()
        r_ell = torch.zeros_like(odds)
        r = [r_ell[:, -1]]
        for _ in range(odds.size(1)):
            r_ell = (log_odds + r_ell).logcumsumexp(1)
            r.append(r_ell[:, -1])
            log_odds, r_ell = log_odds[:, 1:], r_ell[:, :-1]
        r = torch.stack(r, 1)
        log_r = r.log_softmax(1)
        if not batch_dims:
            return log_r.flatten()
        else:
            return log_r.expand(batch_shape + log_r.size()[1:])

    @property
    def mean(self):
        return self.probs.sum(-1)

    @property
    def variance(self):
        return (self.probs * (1.0 - self.probs)).sum(-1)

    @property
    def param_shape(self):
        return self._param.size()

    def expand(self, batch_shape, _instance=None):
        # FIXME(sdrobert): I dunnae ken to do this without using the
        # method _get_checked_instance and _validate_args
        new = self._get_checked_instance(PoissonBinomial, _instance)
        batch_shape = list(batch_shape)
        if "total_count" in self.__dict__:
            new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape + [self.probs.size(-1)])
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape + [self.logits.size(-1)])
            new._param = new.logits
        if "odds" in self.__dict__:
            new.odds = self.odds.expand(batch_shape + [self.odds.size(-1)])
            new._param = new.odds
        if "log_r" in self.__dict__:
            new.log_r = self.r.expand(batch_shape + [self.log_r.size(-1)])
        super(PoissonBinomial, new).__init__(
            torch.Size(batch_shape), validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        odds = self.odds
        odds = odds.expand(shape + odds.size()[-1:])
        with torch.no_grad():
            return poisson_binomial(odds)

    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        shape = value.shape
        log_r = self.log_r
        log_r = log_r.expand(shape + log_r.size()[-1:])
        return log_r.gather(-1, value.long().unsqueeze(-1)).squeeze(-1)


@torch.jit.script_if_tracing
@torch.no_grad()
def extended_conditional_bernoulli(
    w_f: torch.Tensor,
    lmax: torch.Tensor,
    kmax: int = 1,
    r_f: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert kmax == 1, "kmax > 1 not yet implemented"
    device = w_f.device
    if w_f.dim() == 2:
        # the user passed regular odds instead of the relevant odds forward.
        nmax, tmax = w_f.size()
        # call checks lmax size
        w_f = extract_relevant_odds_forward(w_f, lmax, True)
    else:
        _, nmax, diffmax = w_f.size()
        tmax = diffmax + int(lmax.min()) - 1
        assert device == lmax.device and lmax.dim() == 1 and lmax.size(0) == nmax
    if r_f is None:
        r_f = R_forward(w_f, lmax, kmax, True, True)
    else:
        assert (
            r_f.dim() == 3
            and r_f.size(0) == w_f.size(0) + 1
            and r_f.size(1) == nmax
            and r_f.size(2) == diffmax
        )
    assert r_f.dim() == 3
    return _extended_conditional_bernoulli_k_eq_1(w_f, lmax.long(), r_f, tmax)


@torch.jit.script
@torch.no_grad()
def _extended_conditional_bernoulli_k_eq_1(
    w_f: torch.Tensor, lmax: torch.Tensor, r_f: torch.Tensor, tmax: int
) -> torch.Tensor:
    device = w_f.device
    lmax_max, nmax, diffmax = w_f.size()
    remainder = lmax
    nrange = torch.arange(nmax)
    drange = torch.arange(diffmax)
    lmax_max = r_f.size(0) - 1
    tau_ellp1 = torch.full((nmax, 1), diffmax)
    b = torch.zeros((nmax, tmax), device=device)
    for _ in range(lmax_max):
        valid_ell = remainder > 0
        mask_ell = (tau_ellp1 < drange) & valid_ell.unsqueeze(1)
        weights_ell = (
            w_f[remainder - 1, nrange] * r_f[remainder - 1, nrange]
        ).masked_fill(mask_ell, 0)
        tau_ell = torch.multinomial(weights_ell + (~valid_ell).unsqueeze(1), 1, True)
        remainder = (remainder - 1).clamp_min_(0)
        b[nrange, (tau_ell.squeeze(1) + remainder).clamp_max_(tmax - 1)] += valid_ell
        tau_ellp1 = tau_ell
    return b[:, :tmax]


class ConditionalBernoulli(torch.distributions.ExponentialFamily):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "given_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
        "odds": constraints.greater_than_eq(0.0),
    }
    _mean_carrier_measure = 0

    class SupportConstraint(constraints.Constraint):
        is_discrete = True
        event_dim = 1

        def __init__(
            self, total_count: torch.Tensor, given_count: torch.Tensor, tmax: int
        ) -> None:
            self.given_count = given_count
            self.total_count_mask = total_count.unsqueeze(-1) >= torch.arange(
                tmax, device=total_count.device
            )
            super().__init__()

        def check(self, value: torch.Tensor) -> torch.Tensor:
            is_bool = ((value == 0) | (value == 1)).all(-1)
            isnt_gte_tc = (self.total_count_mask.expand_as(value) * value).sum(-1) != 0
            value_sum = value.sum(-1)
            matches_count = value_sum == self.given_count.expand_as(value_sum)
            return is_bool & isnt_gte_tc & matches_count

    def __init__(
        self,
        given_count: Union[torch.Tensor, int],
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        odds: Optional[torch.Tensor] = None,
        total_count: Union[torch.Tensor, int, None] = None,
        validate_args=None,
    ):
        non_none = {x for x in (probs, logits, odds) if x is not None}
        if len(non_none) != 1:
            raise ValueError("Exactly one of probs, logits, or must be non-None.")
        param = non_none.pop()
        if param.dim() < 1:
            raise ValueError("parameter must be at least 1-dimensional")
        param_size = param.size()
        batch_size = param_size[:-1]
        tmax = param_size[-1]
        if total_count is None:
            mask = None
        else:
            self.total_count = (
                torch.as_tensor(total_count).expand(batch_size).type_as(param)
            )
            mask = self.total_count.unsqueeze(-1) <= torch.arange(tmax)
        if probs is not None:
            self.probs = self._param = (
                probs if mask is None else probs.masked_fill(mask, 0.0)
            )
        elif logits is not None:
            self.logits = self._param = (
                logits if mask is None else logits.masked_fill(mask, -float("inf"))
            )
        else:
            self.odds = self._param = (
                odds if mask is None else odds.masked_fill(mask, 0.0)
            )
        self.given_count = (
            torch.as_tensor(given_count).expand(batch_size).type_as(param)
        )
        super(ConditionalBernoulli, self).__init__(
            batch_size, param_size[-1:], validate_args
        )

    @lazy_property
    def total_count(self):
        return torch.full(self.batch_shape, self.odds.size(-1)).type_as(self._param)

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return self.SupportConstraint(
            self.total_count, self.given_count, self._event_shape[0]
        )

    @lazy_property
    def probs(self):
        return self.odds / (1 + self.odds)

    @lazy_property
    def odds(self):
        return self.logits.exp()

    @lazy_property
    def logits(self):
        return self.probs.logit()

    @lazy_property
    def odds_f(self):
        odds, given_count = self.odds, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            odds, given_count = odds.unqueeze(0), given_count.view(1)
        else:
            odds, given_count = (
                odds.flatten(end_dim=batch_dims - 1),
                given_count.flatten(),
            )
        odds_f = extract_relevant_odds_forward(odds, given_count, True)
        if not batch_dims:
            odds_f = odds_f.squeeze(1)
        else:
            odds_f = odds_f.reshape([odds_f.size(0)] + batch_shape + [odds_f.size(-1)])
        return odds_f

    @lazy_property
    def r_f(self):
        odds_f, given_count = self.odds_f, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            odds_f, given_count = odds_f.unqueeze(0), given_count.view(1)
        else:
            odds_f, given_count = (
                odds_f.flatten(1, batch_dims),
                given_count.flatten(),
            )
        r_f = R_forward(odds_f, given_count, 1, True, True)
        if not batch_dims:
            r_f = r_f.squeeze(1)
        else:
            r_f = r_f.reshape([r_f.size(0)] + batch_shape + [r_f.size(-1)])
        return r_f

    @lazy_property
    def log_partition(self):
        r_f, given_count = self.r_f, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            r_f, given_count = r_f.unqueeze(0), given_count.view(1)
        else:
            r_f, given_count = (
                r_f.flatten(1, batch_dims),
                given_count.flatten(),
            )
        given_count = given_count.long()
        _, nmax, diffdim = r_f.size()
        return (
            r_f[
                given_count,
                torch.arange(nmax),
                diffdim + given_count.min() - given_count - 1,
            ]
            .log()
            .view(batch_shape)
        )

    @property
    def mean(self):
        return self.probs.sum(-1)

    @property
    def variance(self):
        return (self.probs * (1.0 - self.probs)).sum(-1)

    @property
    def param_shape(self):
        return self._param.size()

    @property
    def _natural_params(self):
        return self.logits

    def _log_normalizer(self, logits):
        assert torch.allclose(logits, self.logits)
        return self.log_partition

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ConditionalBernoulli, _instance)
        batch_shape = list(batch_shape)
        new.given_count = self.given_count.expand(batch_shape)
        if "total_count" in self.__dict__:
            new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape + [self.probs.size(-1)])
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape + [self.logits.size(-1)])
            new._param = new.logits
        if "odds" in self.__dict__:
            new.odds = self.odds.expand(batch_shape + [self.odds.size(-1)])
            new._param = new.odds
        if "odds_f" in self.__dict__:
            new.odds_f = self.r_f.expand(
                [self.w_f.size(0)] + batch_shape + [self.w_f.size(-1)]
            )
        if "r_f" in self.__dict__:
            new.r_f = self.r.expand(
                [self.r_f.size(0)] + batch_shape + [self.r_f.size(-1)]
            )
        if "log_partition" in self.__dict__:
            new.log_partition = self.log_partition.expand(batch_shape)
        super(ConditionalBernoulli, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            odds_f = (
                self.odds_f.expand(sample_shape + self.odds_f.shape)
                .transpose(0, 1)
                .flatten(1, -2)
            )
            r_f = (
                self.r_f.expand(sample_shape + self.r_f.shape)
                .transpose(0, 1)
                .flatten(1, -2)
            )
            given_count = self.given_count.expand(shape[:-1]).flatten()
            b = extended_conditional_bernoulli(odds_f, given_count, 1, r_f)
        return b.view(shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        num = (value * self.logits.expand_as(value)).sum(-1)
        denom = self.log_partition.expand_as(num)
        return num - denom


# TESTS


def test_poisson_binomial():
    # The Poisson Binomial equals the Binomial distribution when all probs are equal
    torch.manual_seed(1)
    tmax, nmax, mmax = 5, 16, 2 ** 15
    p = torch.rand(nmax, requires_grad=True)
    total_count = torch.randint(tmax + 1, (nmax,))

    binom = torch.distributions.Binomial(total_count, probs=p)

    p_ = p.unsqueeze(1).expand(nmax, tmax)
    poisson_binom = PoissonBinomial(total_count, probs=p_)
    lmax = poisson_binom.sample([mmax])
    mean = lmax.mean(0)
    assert torch.allclose(mean, p * total_count, atol=1e-2)

    log_prob_act = poisson_binom.log_prob(lmax).t()
    (grad_log_prob_act,) = torch.autograd.grad(log_prob_act.mean(), [p])

    binom = torch.distributions.Binomial(total_count, probs=p)
    log_prob_exp = binom.log_prob(lmax).t()
    assert torch.allclose(log_prob_exp, log_prob_act, atol=1e-3)
    (grad_log_prob_exp,) = torch.autograd.grad(log_prob_exp.mean(), [p])
    assert torch.allclose(grad_log_prob_exp, grad_log_prob_act, atol=1e-3)


def test_conditional_bernoulli():
    torch.manual_seed(2)
    tmax, nmax, mmax = 32, 8, 2 ** 15
    logits = torch.randn(nmax, tmax, requires_grad=True)
    idx_1_first = []
    idx_1_second = []
    for first in range(tmax - 1):
        for second in range(first + 1, tmax):
            idx_1_first.append(first)
            idx_1_second.append(second)
    idx_1_first = torch.tensor(idx_1_first)
    idx_1_second = torch.tensor(idx_1_second)
    logits_ = logits[:, idx_1_first] + logits[:, idx_1_second]
    probs_ = logits_.softmax(1)  # (nmax, s)
    first_matches = torch.arange(tmax).unsqueeze(1) == idx_1_first  # (tmax, s)
    second_matches = torch.arange(tmax).unsqueeze(1) == idx_1_second  # (tmax, s)
    inclusions_exp = (probs_.unsqueeze(1) * (first_matches | second_matches)).sum(2)
    assert torch.allclose(inclusions_exp.sum(1), torch.tensor(2.0))
    conditional_bernoulli = ConditionalBernoulli(2, logits=logits)

    b = conditional_bernoulli.sample([mmax])
    assert b.shape == torch.Size([mmax, nmax, tmax])
    assert (b.sum(-1) == 2).all()
    assert (b.max(-1)[0] == 1).all()
    inclusions_act = b.mean(0)
    assert torch.allclose(inclusions_exp, inclusions_act, atol=1e-2)

    poisson_binomial = PoissonBinomial(tmax, logits=logits)
    log_prob_act = conditional_bernoulli.log_prob(b) + poisson_binomial.log_prob(
        torch.full((nmax,), 2)
    )
    (grad_log_prob_act,) = torch.autograd.grad(log_prob_act.mean(), [logits])

    bernoulli = torch.distributions.Bernoulli(logits=logits)
    log_prob_exp = bernoulli.log_prob(b).sum(-1)
    assert log_prob_exp.shape == log_prob_act.shape
    assert torch.allclose(log_prob_exp, log_prob_act)
    (grad_log_prob_exp,) = torch.autograd.grad(log_prob_exp.mean(), [logits])
    assert torch.allclose(grad_log_prob_exp, grad_log_prob_act)

    total_count = torch.randint(tmax, (nmax,), dtype=torch.float)
    given_count = (torch.rand((nmax,)) * (total_count + 1)).floor_()
    conditional_bernoulli = ConditionalBernoulli(
        given_count, logits=logits, total_count=total_count, validate_args=True
    )
    b = conditional_bernoulli.sample([mmax])
    total_count_mask = total_count.unsqueeze(1) > torch.arange(tmax)
    assert (b.sum(-1) == given_count.unsqueeze(0)).all()
    assert ((b * total_count_mask).sum(-1) == given_count.unsqueeze(0)).all()
    assert (b.max(-1)[0] <= 1).all()


# XXX(sdrobert): The following did not work. I've had difficulty with algorithms based
# on subtraction/division.
# @torch.no_grad()
# def extended_conditional_bernoulli(
#     w_f: torch.Tensor,
#     lmax: torch.Tensor,
#     kmax: int = 1,
#     r_f: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     assert kmax == 1, "kmax > 1 not yet implemented"
#     device = w_f.device
#     if w_f.dim() == 2:
#         # the user passed regular odds instead of the relevant odds forward.
#         nmax, tmax = w_f.size()
#         # call checks lmax size
#         w_f = extract_relevant_odds_forward(w_f, lmax, True)
#     else:
#         _, nmax, diffmax = w_f.size()
#         tmax = diffmax + int(lmax.min()) - 1
#         assert device == lmax.device and lmax.dim() == 1 and lmax.size(0) == nmax
#     if r_f is None:
#         r_f = R_forward(w_f, lmax, kmax, True, True)
#     else:
#         assert (
#             r_f.dim() == 3
#             and r_f.size(0) == w_f.size(0) + 1
#             and r_f.size(1) == nmax
#             and r_f.size(2) == diffmax
#         )
#     # "Direct Sampling" (Procedure 5) from Chen and Liu '97, modified for the
#     # interpretations of w_f and r_f. Full tables of w_f and r_f (including irrelevant
#     # values) would always involve moving left (i.e. subtracting 1 from idx_0 below at
#     # each iteration) and optionally up (i.e. subtracting 1 from idx_2) depending on
#     # whether or not b_t was drawn to be high. Hence, we start at the bottom-right
#     # element and take steps either left or up-left to the top-left. However, the
#     # relevant tables w_f and r_f exclude all irrelevant values outside of a shifting
#     # window on idx_2. Effectively, a movement up in the relevant-only w_f and r_f
#     # corresponds to an up-left movement on the full table. Hence, we move either up
#     # or left (not up-left) in the below loop over relevant values.
#     b = torch.empty((tmax, nmax), device=device)
#     idx_0 = lmax.long()  # a.k.a. how many samples left to draw, ell
#     idx_1 = torch.arange(nmax, device=device)  # batch dim
#     idx_2 = tmax - idx_0  # a.k.a. (prefix size - ell)
#     u_k = torch.rand((nmax,), device=device) * r_f[idx_0, idx_1, idx_2]
#     for t in range(tmax - 1, -1, -1):
#         w_k = w_f[(idx_0 - 1).clamp_min_(0), idx_1, idx_2]
#         r_k = r_f[idx_0, idx_1, (idx_2 - 1).clamp_min_(0)]
#         b_t = u_k >= r_k
#         b[t] = b_t
#         u_k = torch.where(b_t, (u_k - r_k) / w_k, u_k)
#         b_t_ = b_t.long()
#         idx_0 -= b_t_
#         idx_2 += b_t_ - 1
#     # print(u_k)
#     assert (idx_0 == 0).all()
#     assert (idx_2 == 0).all()
#     return b.t()
