"""Distributions, in particular the Conditional Bernoulli"""

import math
from forward_backward import (
    R_forward,
    extract_relevant_odds_forward,
    log_R_forward,
)
import config
from typing import Optional, Union
import torch

import torch.distributions.constraints as constraints
from torch.distributions.utils import lazy_property, logits_to_probs


class PoissonBinomial(torch.distributions.Distribution):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }

    def __init__(
        self,
        total_count: Union[torch.Tensor, int, None] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args=None,
    ):
        non_none = {x for x in (probs, logits) if x is not None}
        if len(non_none) != 1:
            raise ValueError("Probs or logits must be non-none, not both")
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
        else:
            self.logits = self._param = (
                logits if mask is None else logits.masked_fill(mask, -float("inf"))
            )
        super(PoissonBinomial, self).__init__(batch_size, validate_args=validate_args)

    @lazy_property
    def total_count(self):
        return torch.full(self.batch_shape, self.odds.size(-1)).type_as(self._param)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @lazy_property
    def probs(self):
        return self.logits.sigmoid()

    @lazy_property
    def logits(self):
        return self.probs.logit()

    @lazy_property
    def log_r(self):
        logits, total_count = self.logits, self.total_count
        batch_shape = self.batch_shape
        batch_dims = len(batch_shape)
        if not batch_dims:
            logits, total_count = logits.unsqueeze(0), total_count.view(1)
        else:
            logits, total_count = (
                logits.flatten(end_dim=batch_dims - 1),
                total_count.flatten(),
            )
        # unfortunately, the sums can get pretty big. This leads to NaNs in the
        # backward pass. We go straight into log here
        log_r_ell = torch.zeros_like(logits)
        log_r = [log_r_ell[:, -1]]
        for _ in range(logits.size(1)):
            log_r_ell = (logits + log_r_ell).logcumsumexp(1)
            log_r.append(log_r_ell[:, -1])
            logits, log_r_ell = logits[:, 1:], log_r_ell[:, :-1]
        log_r = torch.stack(log_r, 1)
        log_r = log_r.log_softmax(1)
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
        if "log_r" in self.__dict__:
            new.log_r = self.r.expand(batch_shape + [self.log_r.size(-1)])
        super(PoissonBinomial, new).__init__(
            torch.Size(batch_shape), validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            probs = self.probs.expand(shape + self.probs.size()[-1:])
            lmax = torch.bernoulli(probs).sum(-1)
        return lmax

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
    log_space: bool = False,
    neg_inf: float = config.EPS_INF,
) -> torch.Tensor:
    assert kmax == 1, "kmax > 1 not yet implemented"
    device = w_f.device
    if w_f.dim() == 2:
        # the user passed regular odds instead of the relevant odds forward.
        nmax, tmax = w_f.size()
        # call checks lmax size
        w_f = extract_relevant_odds_forward(
            w_f, lmax, True, neg_inf if log_space else 0
        )
    else:
        _, nmax, diffmax = w_f.size()
        tmax = diffmax + int(lmax.min()) - 1
        assert device == lmax.device and lmax.dim() == 1 and lmax.size(0) == nmax
    if r_f is None:
        r_f = (
            log_R_forward(w_f, lmax, kmax, True, True)
            if log_space
            else R_forward(w_f, lmax, kmax, True, True)
        )
    else:
        assert (
            r_f.dim() == 3
            and r_f.size(0) == w_f.size(0) + 1
            and r_f.size(1) == nmax
            and r_f.size(2) == diffmax
        )
    assert r_f.dim() == 3
    if log_space:
        return _log_extended_conditional_bernoulli_k_eq_1(
            w_f, lmax.long(), r_f, tmax, neg_inf
        )
    else:
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


@torch.jit.script
@torch.no_grad()
def _log_extended_conditional_bernoulli_k_eq_1(
    logits_f: torch.Tensor,
    lmax: torch.Tensor,
    log_r_f: torch.Tensor,
    tmax: int,
    neg_inf: float = config.EPS_INF,
) -> torch.Tensor:
    assert neg_inf > -float("inf")
    device = logits_f.device
    lmax_max, nmax, diffmax = logits_f.size()
    remainder = lmax
    nrange = torch.arange(nmax)
    drange = torch.arange(diffmax)
    lmax_max = log_r_f.size(0) - 1
    tau_ellp1 = torch.full((nmax, 1), diffmax)
    b = torch.zeros((nmax, tmax), device=device)
    for _ in range(lmax_max):
        valid_ell = remainder > 0
        mask_ell = (tau_ellp1 < drange) & valid_ell.unsqueeze(1)
        log_weights_ell = (
            logits_f[remainder - 1, nrange] + log_r_f[remainder - 1, nrange]
        ).masked_fill(mask_ell, -float("inf"))
        log_weights_ell -= log_weights_ell.max(1, keepdim=True)[0].clamp_min_(neg_inf)
        tau_ell = torch.multinomial(
            log_weights_ell.exp() + (~valid_ell).unsqueeze(1), 1, True
        )
        remainder = (remainder - 1).clamp_min_(0)
        b[nrange, (tau_ell.squeeze(1) + remainder).clamp_max_(tmax - 1)] += valid_ell
        tau_ellp1 = tau_ell
    return b[:, :tmax]


class ConditionalBernoulliConstraint(constraints.Constraint):
    is_discrete = True
    event_dim = 1

    def __init__(
        self, total_count: torch.Tensor, given_count: torch.Tensor, tmax: int
    ) -> None:
        self.given_count = given_count
        self.total_count_mask = total_count.unsqueeze(-1) <= torch.arange(
            tmax, device=total_count.device
        )
        super().__init__()

    def check(self, value: torch.Tensor) -> torch.Tensor:
        is_bool = ((value == 0) | (value == 1)).all(-1)
        isnt_gte_tc = (self.total_count_mask.expand_as(value) * value).sum(-1) == 0
        value_sum = value.sum(-1)
        matches_count = value_sum == self.given_count.expand_as(value_sum)
        return is_bool & isnt_gte_tc & matches_count


class ConditionalBernoulli(torch.distributions.ExponentialFamily):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "given_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }
    _mean_carrier_measure = 0

    def __init__(
        self,
        given_count: Union[torch.Tensor, int],
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        total_count: Union[torch.Tensor, int, None] = None,
        validate_args=None,
    ):
        non_none = {x for x in (probs, logits) if x is not None}
        if len(non_none) != 1:
            raise ValueError("Exactly one of probs or logits must be non-None.")
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
        else:
            self.logits = self._param = (
                logits if mask is None else logits.masked_fill(mask, -float("inf"))
            )
        self.given_count = (
            torch.as_tensor(given_count).expand(batch_size).type_as(param)
        )
        super(ConditionalBernoulli, self).__init__(
            batch_size, param_size[-1:], validate_args
        )

    @lazy_property
    def total_count(self):
        return torch.full(self.batch_shape, self._param.size(-1)).type_as(self._param)

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return ConditionalBernoulliConstraint(
            self.total_count, self.given_count, self._event_shape[0]
        )

    @lazy_property
    def probs(self):
        return self.logits.sigmoid()

    @lazy_property
    def logits(self):
        return self.probs.logit()

    @lazy_property
    def logits_f(self):
        logits, given_count = self.logits, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            logits, given_count = logits.unsqueeze(0), given_count.view(1)
        else:
            logits, given_count = (
                logits.flatten(end_dim=batch_dims - 1),
                given_count.flatten(),
            )
        logits_f = extract_relevant_odds_forward(
            logits, given_count, True, -float("inf")
        )
        if not batch_dims:
            logits_f = logits_f.squeeze(1)
        else:
            logits_f = logits_f.reshape(
                [logits_f.size(0)] + batch_shape + [logits_f.size(-1)]
            )
        return logits_f

    @lazy_property
    def log_r_f(self):
        logits_f, given_count = self.logits_f, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            logits_f, given_count = logits_f.unsqueeze(1), given_count.view(1)
        else:
            logits_f, given_count = (
                logits_f.flatten(1, batch_dims),
                given_count.flatten(),
            )
        log_r_f = log_R_forward(logits_f, given_count, 1, True, True)
        if not batch_dims:
            log_r_f = log_r_f.squeeze(1)
        else:
            log_r_f = log_r_f.reshape(
                [log_r_f.size(0)] + batch_shape + [log_r_f.size(-1)]
            )
        return log_r_f

    @property
    def log_partition(self):
        log_r_f, given_count = self.log_r_f, self.given_count
        batch_shape = list(self.batch_shape)
        batch_dims = len(batch_shape)
        if not batch_dims:
            log_r_f, given_count = log_r_f.unsqueeze(0), given_count.view(1)
        else:
            log_r_f, given_count = (
                log_r_f.flatten(1, batch_dims),
                given_count.flatten(),
            )
        given_count = given_count.long()
        _, nmax, diffdim = log_r_f.size()
        return log_r_f[
            given_count,
            torch.arange(nmax),
            diffdim + given_count.min() - given_count - 1,
        ].view(batch_shape)

    @property
    def mean(self):
        # inclusion probabilities
        raise NotImplementedError

    @property
    def variance(self):
        return self.mean * (1 - self.mean)

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
        if "logits_f" in self.__dict__:
            new.logits_f = self.r_f.expand(
                [self.w_f.size(0)] + batch_shape + [self.w_f.size(-1)]
            )
        if "log_r_f" in self.__dict__:
            new.log_r_f = self.r.expand(
                [self.log_r_f.size(0)] + batch_shape + [self.log_r_f.size(-1)]
            )
        super(ConditionalBernoulli, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            logits_f, log_r_f, lmax = self.logits_f, self.log_r_f, self.given_count
            if not len(self.batch_shape):
                logits_f, log_r_f = logits_f.unsqueeze(1), log_r_f.unsqueeze(1)
                lmax = lmax.unsqueeze(0)
            if sample_shape:
                logits_f = (
                    logits_f.unsqueeze(1)
                    .expand(logits_f.shape[:1] + sample_shape + logits_f.shape[1:])
                    .flatten(1, 2)
                )
                log_r_f = (
                    log_r_f.unsqueeze(1)
                    .expand(log_r_f.shape[:1] + sample_shape + log_r_f.shape[1:])
                    .flatten(1, 2)
                )
                lmax = lmax.expand(shape[:-1]).flatten()
            b = extended_conditional_bernoulli(logits_f, lmax, 1, log_r_f, True)
        return b.view(shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        num = (value * self.logits.clamp_min(config.EPS_INF).expand_as(value)).sum(-1)
        denom = self.log_partition.expand_as(num)
        return num - denom


@torch.jit.script
@torch.no_grad()
def simple_sampling_without_replacement(
    tmax: torch.Tensor, lmax: torch.Tensor, tmax_max: Optional[int] = None
) -> torch.Tensor:
    if tmax_max is None:
        tmax_max = int(tmax.max().item())
    tmax, lmax = torch.broadcast_tensors(tmax, lmax)
    b = torch.empty(torch.Size([tmax_max]) + tmax.shape)
    remainder_ell = lmax
    remainder_t = tmax
    for t in range(tmax_max):
        p = remainder_ell / remainder_t
        b_t = torch.bernoulli(p)
        b[t] = b_t
        remainder_ell = remainder_ell - b_t
        remainder_t = (remainder_t - 1).clamp_min_(1)
    return b.movedim(0, -1)


class SimpleSamplingWithoutReplacement(torch.distributions.ExponentialFamily):

    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "given_count": constraints.nonnegative_integer,
    }
    _mean_carrier_measure = 0

    def __init__(
        self,
        given_count: Union[int, torch.Tensor],
        total_count: Union[int, torch.Tensor],
        out_size: Optional[int] = None,
        validate_args=None,
    ):
        given_count = torch.as_tensor(given_count)
        total_count = torch.as_tensor(total_count)
        if out_size is None:
            out_size = total_count.max()
        given_count, total_count = torch.broadcast_tensors(given_count, total_count)
        batch_shape = given_count.size()
        event_shape = torch.Size([out_size])
        self.total_count, self.given_count = total_count, given_count
        super(SimpleSamplingWithoutReplacement, self).__init__(
            batch_shape, event_shape, validate_args
        )

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return ConditionalBernoulliConstraint(
            self.total_count, self.given_count, self.event_shape[0]
        )

    @lazy_property
    def log_partition(self) -> torch.Tensor:
        # log total_count choose given_count
        log_factorial = (
            torch.arange(
                1,
                self.event_shape[0] + 1,
                device=self.total_count.device,
                dtype=torch.float,
            )
            .log()
            .cumsum(0)
        )
        t_idx = (self.total_count.long() - 1).clamp_min(0)
        g_idx = (self.given_count.long() - 1).clamp_min(0)
        tmg_idx = (self.total_count.long() - self.given_count.long() - 1).clamp_min(0)
        return log_factorial[t_idx] - log_factorial[g_idx] - log_factorial[tmg_idx]

    @property
    def mean(self):
        len_mask = self.total_count.unsqueeze(-1) <= torch.arange(self.event_shape[0])
        return (
            (self.given_count / self.total_count.clamp_min(1.0))
            .unsqueeze(-1)
            .expand(self.batch_shape + self.event_shape)
        ).masked_fill(len_mask, 0.0)

    @property
    def variance(self):
        return self.mean * (1 - self.mean)

    @property
    def _natural_params(self):
        return self.zeros(self.batch_shape + self.event_shape)

    def _log_normalizer(self, logits):
        assert torch.allclose(logits, torch.tensor(0))
        return self.log_partition

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SimpleSamplingWithoutReplacement, _instance)
        batch_shape = list(batch_shape)
        new.given_count = self.given_count.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)

        if "log_partition" in self.__dict__:
            new.log_partition = self.log_partition.expand(batch_shape)

        super(SimpleSamplingWithoutReplacement, new).__init__(
            torch.Size(batch_shape), self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            total_count = self.total_count.expand(shape[:-1])
            given_count = self.given_count.expand(shape[:-1])
            b = simple_sampling_without_replacement(
                total_count, given_count, self.event_shape[0]
            )
        return b

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-self.log_partition).expand(value.shape[:-1])


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

    binom = torch.distributions.Binomial(total_count, probs=p.requires_grad_(True))
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
    assert ((b == 0.0) | (b == 1.0)).all()
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


def test_simple_sampling_without_replacement():
    torch.manual_seed(3)
    tmax_max, nmax, mmax = 16, 8, 2 ** 15
    tmax = torch.randint(tmax_max + 1, size=(nmax,), dtype=torch.float)
    lmax = (torch.rand(nmax) * (tmax + 1)).floor_()

    sswor = SimpleSamplingWithoutReplacement(lmax, tmax, tmax_max, True)
    b = sswor.sample([mmax])
    assert ((b == 0.0) | (b == 1.0)).all()
    assert (b.sum(-1) == lmax).all()
    tmax_mask = tmax.unsqueeze(1) > torch.arange(tmax_max)
    b = b * tmax_mask
    assert (b.sum(-1) == lmax).all()
    assert torch.allclose(b.mean(0), sswor.mean, atol=1e-2)

    lp_exp = []
    for n in range(nmax):
        tmax_n, lmax_n = int(tmax[n].item()), int(lmax[n].item())
        lp_exp.append(
            math.log(
                (math.factorial(tmax_n - lmax_n) * math.factorial(lmax_n))
                / math.factorial(tmax_n)
            )
        )
    lp_exp = torch.tensor(lp_exp).expand(mmax, nmax)
    lp_act = sswor.log_prob(b)
    assert torch.allclose(lp_exp, lp_act)


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
