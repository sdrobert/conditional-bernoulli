# Copyright 2021 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributions, in particular the Conditional Bernoulli"""

from forward_backward import (
    R_forward,
    extract_relevant_odds_forward,
    log_R_forward,
)
import config
from typing import Optional, Union

import torch

import pydrobert.torch.distributions as _d
import pydrobert.torch.functional as _f
import torch.distributions.constraints as constraints
from torch.distributions.utils import lazy_property


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
        # FIXME(anon): I dunnae ken to do this without using the method
        # _get_checked_instance and _validate_args
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
        ).masked_fill_(mask_ell, -float("inf"))
        norm = log_weights_ell.max(1, keepdim=True)[0]
        log_weights_ell = (log_weights_ell - norm).masked_fill_(norm.isinf(), neg_inf)
        tau_ell = torch.multinomial(
            log_weights_ell.exp_() + (~valid_ell).unsqueeze(1), 1, True
        )
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

    @property
    def has_enumerate_support(self) -> bool:
        return (self.given_count == self.given_count.flatten()[0]).all().item()

    @lazy_property
    def total_count(self):
        return torch.full(self.batch_shape, self._event_shape[0]).type_as(self._param)

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return _d.BinaryCardinalityConstraint(
            self.total_count, self.given_count, self._event_shape[0]
        )

    def enumerate_support(self, expand=True) -> torch.Tensor:
        if not self.has_enumerate_support:
            raise NotImplementedError(
                "given_count must all be equal to enumerate support"
            )
        total = self._event_shape[0]
        given = int(self.given_count.flatten()[0].item())
        support = _f.enumerate_binary_sequences_with_cardinality(total, given)
        support = support.view((-1,) + (1,) * len(self.batch_shape) + (total,))
        if expand:
            support = support.expand((-1,) + self.batch_shape + (total,))
        return support

    @lazy_property
    def probs(self):
        return self.logits.sigmoid()

    @lazy_property
    def logits(self):
        return self.probs.logit()

    @lazy_property
    def logits_f(self):
        logits, given_count = self.logits, self.given_count
        batch_dims = len(self.batch_shape)
        if not batch_dims:
            logits, given_count = logits.unsqueeze(0), given_count.view(1)
        else:
            logits = logits.flatten(end_dim=-2)
            given_count = given_count.flatten()
        logits_f = extract_relevant_odds_forward(
            logits, given_count, True, -float("inf")
        )
        if not batch_dims:
            logits_f = logits_f.squeeze(1)
        else:
            logits_f = logits_f.unflatten(1, self.batch_shape)
        return logits_f

    @lazy_property
    def log_r_f(self):
        logits_f, given_count = self.logits_f, self.given_count
        batch_dims = len(self.batch_shape)
        if not batch_dims:
            logits_f, given_count = logits_f.unsqueeze(1), given_count.view(1)
        else:
            logits_f = logits_f.flatten(1, batch_dims)
            given_count = given_count.flatten()
        log_r_f = log_R_forward(logits_f, given_count, 1, True, True)
        if not batch_dims:
            log_r_f = log_r_f.squeeze(1)
        else:
            log_r_f = log_r_f.unflatten(1, self.batch_shape)
        return log_r_f

    @property
    def log_partition(self):
        log_r_f = self.log_r_f[..., -1]
        given_count = self.given_count.long()
        given_count = given_count.unsqueeze(0)
        given_count = given_count.expand((1,) + log_r_f.shape[1:])
        return log_r_f.gather(0, given_count).squeeze(0)

    @property
    @torch.no_grad()
    def mean(self):
        # inclusion probabilities
        log_r_f, logits_f, logits = self.log_r_f, self.logits_f, self.logits
        device = log_r_f.device
        given_count = self.given_count
        batch_dims = len(self.batch_shape)
        if not batch_dims:
            log_r_f, given_count = log_r_f.unsqueeze(1), given_count.view(1)
            logits, logits_f = logits.unsqueeze(0), logits_f.unsqueeze(1)
        else:
            log_r_f = log_r_f.flatten(1, batch_dims)
            logits_f = logits_f.flatten(1, batch_dims)
            given_count = given_count.flatten()
        given_count = given_count.long()
        lmax_max_p1, nmax, diffdim = log_r_f.shape
        lmax_max = lmax_max_p1 - 1
        tmax = self.event_shape[0]
        assert logits_f.shape == (lmax_max, nmax, diffdim)
        li = torch.full((lmax_max, nmax, tmax), -float("inf"), device=device)
        keep = torch.full((lmax_max, tmax), 0, device=device, dtype=torch.bool)
        li_ = []
        logits = logits.flip([1])
        nrange = torch.arange(nmax, device=device)
        for ell in range(lmax_max):
            remainder = given_count - ell - 1
            invalid = remainder < 0
            right = min(ell + diffdim, tmax)
            keep[ell, ell:right] = True
            li_.append(
                (
                    logits_f[ell, :, : right - ell] + log_r_f[ell, :, : right - ell]
                ).flatten()
            )
            if ell:
                cur_r_b = torch.cat(
                    [
                        torch.full((nmax, ell), -float("inf"), device=device),
                        logits[:, ell:],
                    ],
                    1,
                )
                cur_r_b[:, 1:] += last_r_b[:, :-1]
                cur_r_b = cur_r_b.logcumsumexp(1)
            else:
                cur_r_b = torch.zeros_like(logits)
            cur_r_b.masked_fill_(invalid.unsqueeze(1), -float("inf"))
            li[remainder.clamp_min_(-1), nrange] = cur_r_b.flip([1])
            last_r_b = cur_r_b
        li_ = torch.cat(li_)
        li2 = torch.full_like(li, -float("inf")).masked_scatter_(keep.unsqueeze(1), li_)
        li2[..., :-1] += li[..., 1:]
        li = li2
        li = li.logsumexp(0) - self.log_partition.view(-1, 1)
        assert (li <= 0).all(), (li > 0).nonzero()
        return li.exp().view(self.batch_shape + (tmax,))

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
            new.log_r_f = self.log_r_f.expand(
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


class DreznerFarnumBernoulli(torch.distributions.Distribution):
    r"""Drezner-Farnum dependent Bernoulli sequences

    .. math::

        P(b_t=1|b_{1:t-1}; \theta_{1,2}) = \begin{cases}
            \sigma(\theta_{1,t}) & t = 1 \\
            (1 - \sigma(\theta_{2,t}))\sigma(\theta_{1,t}) +
                \sigma(\theta_{2,t})\frac{\sum_{t'=1}^{t - 1}b_t}{t - 1} & t > 1
        \end{cases} \\
        \sigma(x) = \frac{1}{1 + \exp(-x)}
    
    Parameters
    ----------
    theta_1
        :math:`\theta_1`. At least 1-dimensional with final dimension :math:`t`.
    theta_2
        :math:`\theta_2`. Broadcasts with `theta_1`.
    """

    theta_1: torch.Tensor
    theta_2: torch.Tensor
    T: int

    arg_constraints = {
        "theta_1": constraints.real,
        "theta_2": constraints.real,
        "T": constraints.positive_integer,
    }

    support = constraints.independent(constraints.boolean, 1)

    def __init__(
        self, theta_1: torch.Tensor, theta_2: torch.Tensor, T: int, validate_args=None
    ):
        self.theta_1 = torch.as_tensor(theta_1)
        self.theta_2 = torch.as_tensor(theta_2)
        self.T = torch.as_tensor(T)
        super().__init__(self.theta_1.shape, torch.Size([T]), validate_args)

    @lazy_property
    def probs(self) -> torch.Tensor:
        return self.theta_1.sigmoid().unsqueeze(-1).expand(self._extended_shape())

    @lazy_property
    def mixture(self) -> torch.Tensor:
        return self.theta_2.sigmoid().unsqueeze(-1).expand(self._extended_shape())

    @property
    def mean(self) -> torch.Tensor:
        return self.probs

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        T, device = self.event_shape[0], self.theta_1.device
        b = torch.empty(shape, device=device)
        u = torch.rand(shape, device=device)
        probs = self.probs.expand(shape)
        mixture = self.mixture.expand(shape)
        b[..., 0] = sum_ = (u[..., 0] < probs[..., 0]).float()
        for tm1 in range(1, T):
            b_t = (
                u[..., tm1]
                < (
                    probs[..., tm1] * (1 - mixture[..., tm1])
                    + mixture[..., tm1] * sum_ / tm1
                )
            ).float()
            sum_ += b_t
            b[..., tm1] = b_t
        return b

    def log_prob(self, b: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(b)
        T, device = b.size(-1), self.theta_1.device
        lp_1 = -torch.nn.functional.binary_cross_entropy_with_logits(
            self.theta_1.unsqueeze(-1).expand_as(b), b, reduction="none"
        )
        lp_11, lp_1 = lp_1[..., 0], lp_1[..., 1:]
        if T == 1:
            return lp_11 + self.theta_2 - self.theta_2.detach()
        lm_2, lm_1 = self.theta_2.sigmoid().log(), (-self.theta_2).sigmoid().log()
        lp_1 = lp_1 + lm_1.unsqueeze(-1)
        p_2 = b[..., :-1].cumsum(-1) / torch.arange(1, T, device=device, dtype=b.dtype)
        p_2 = b[..., 1:] * p_2 + (1 - b[..., 1:]) * (1 - p_2)
        lp_2 = p_2.log() + lm_2.unsqueeze(-1)
        max_, min_ = torch.max(lp_1, lp_2), torch.min(lp_1, lp_2)
        lp = max_ + (min_ - max_).exp().log1p()
        return lp_11 + lp.sum(-1)


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
    tmax, nmax, mmax = 10, 10, 2 ** 15
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

    # assert torch.allclose(conditional_bernoulli.mean.sum(1), torch.tensor(2.0))
    # assert torch.allclose(inclusions_exp, conditional_bernoulli.mean)

    b = conditional_bernoulli.sample([mmax])
    assert b.shape == torch.Size([mmax, nmax, tmax])
    assert (b.sum(-1) == 2).all()
    assert ((b == 0.0) | (b == 1.0)).all()
    inclusions_act = b.mean(0)
    assert torch.allclose(inclusions_exp, inclusions_act, atol=1e-2)

    poisson_binomial = PoissonBinomial(tmax, logits=logits)
    log_prob_act = conditional_bernoulli.log_prob(b) + poisson_binomial.log_prob(
        torch.tensor(2.0).expand((nmax,))
    )
    (grad_log_prob_act,) = torch.autograd.grad(log_prob_act.mean(), [logits])

    bernoulli = torch.distributions.Bernoulli(logits=logits)
    log_prob_exp = bernoulli.log_prob(b).sum(-1)
    assert log_prob_exp.shape == log_prob_act.shape
    assert torch.allclose(log_prob_exp, log_prob_act)
    (grad_log_prob_exp,) = torch.autograd.grad(log_prob_exp.mean(), [logits])
    assert torch.allclose(grad_log_prob_exp, grad_log_prob_act)

    total_count = torch.randint(1, tmax + 1, (nmax,), dtype=torch.float)
    given_count = (torch.rand((nmax,)) * (total_count + 1)).floor_()
    assert (given_count <= total_count).all()
    conditional_bernoulli = ConditionalBernoulli(
        given_count, logits=logits, total_count=total_count, validate_args=True
    )
    b = conditional_bernoulli.sample([mmax])
    total_count_mask = total_count.unsqueeze(1) > torch.arange(tmax)
    assert (b.sum(-1) == given_count.unsqueeze(0)).all()
    assert ((b * total_count_mask).sum(-1) == given_count.unsqueeze(0)).all()
    assert (b.max(-1)[0] <= 1).all()
    
    inclusions_act = conditional_bernoulli.mean
    assert torch.allclose(inclusions_act.sum(1), given_count)
    for n in range(nmax):
        total_count_n = int(total_count[n].item())
        given_count_n = int(given_count[n].item())
        inclusions_act_n = inclusions_act[n]
        assert (inclusions_act_n[total_count_n:] == 0).all()
        logits_n = logits[n, :total_count_n]
        conditional_bernoulli_n = ConditionalBernoulli(
            given_count_n, logits=logits_n, validate_args=True
        )

        assert conditional_bernoulli_n.has_enumerate_support
        b = conditional_bernoulli_n.enumerate_support()
        lpb_exp = torch.distributions.Bernoulli(logits=logits_n).log_prob(b).sum(1)
        lpb_exp -= lpb_exp.logsumexp(0)
        lpb_act = conditional_bernoulli_n.log_prob(b)
        assert torch.allclose(lpb_exp, lpb_act)

        pb = lpb_act.exp()
        assert torch.allclose(pb.sum(), torch.tensor(1.0))
        inclusions_exp_n = (pb.unsqueeze(1) * b).sum(0)
        assert torch.allclose(inclusions_exp_n, inclusions_act_n[:total_count_n])


def test_drezner_farnum():
    torch.manual_seed(3)
    mmax, tmax = 2 ** 15, 5
    theta_1 = torch.randn(1, requires_grad=True)
    theta_2 = torch.randn(1, requires_grad=True)
    zero = torch.zeros(1)
    dist = DreznerFarnumBernoulli(theta_1, theta_2, tmax)
    sample = dist.sample([mmax])
    assert torch.allclose(sample.mean(0), dist.mean, atol=1e-2)
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean() / tmax
    assert torch.isfinite(Elp_act)
    g_theta_1_act, g_theta_2_act = torch.autograd.grad(Elp_act, [theta_1, theta_2])
    assert torch.isclose(g_theta_1_act, zero, atol=1e-2)
    assert torch.isclose(g_theta_2_act, zero, atol=1e-2)

    # make bernoulli draws entirely independent
    theta_2 = -float("inf")
    dist = DreznerFarnumBernoulli(theta_1, theta_2, tmax)
    sample = dist.sample([mmax])
    assert torch.allclose(sample.mean(0), dist.mean, atol=1e-2)
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean() / tmax
    (g_theta_1,) = torch.autograd.grad(Elp_act, [theta_1])
    assert torch.isclose(g_theta_1, zero, atol=1e-2)
    p, omp = theta_1.sigmoid(), (-theta_1).sigmoid()
    Elp_exp = p.detach() * p.log() + omp.detach() * omp.log()
    assert torch.isclose(Elp_exp, Elp_act, atol=1e-2)
    (g_theta_1,) = torch.autograd.grad(Elp_exp, [theta_1])
    assert torch.isclose(g_theta_1, zero, atol=1e-2)

    # now make them entirely dependent
    theta_2 = float("inf")
    dist = DreznerFarnumBernoulli(theta_1, theta_2, tmax)
    sample = dist.sample([mmax])
    assert torch.allclose(sample.mean(0), dist.mean, atol=1e-2)
    assert (sample[..., :1] == sample[..., 1:]).all()
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean()
    (g_theta_1,) = torch.autograd.grad(Elp_act, [theta_1])
    assert torch.isclose(g_theta_1, zero, atol=1e-2)
    assert torch.isclose(Elp_exp, Elp_act, atol=1e-2)


# XXX(anon): The following did not work. I've had difficulty with algorithms based
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
