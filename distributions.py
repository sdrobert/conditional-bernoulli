# Copyright 2022 Sean Robertson

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


from typing import Optional, Union

import torch

import pydrobert.torch.distributions as _d
import pydrobert.torch.functional as _f
import pydrobert.torch.config as config
import torch.distributions.constraints as constraints
from torch.distributions.utils import lazy_property, probs_to_logits, logits_to_probs

from forward_backward import (
    R_forward,
    extract_relevant_odds_forward,
    log_R_forward,
)


class PoissonBinomial(torch.distributions.Distribution):

    arg_constraints = {
        "probs": constraints.independent(constraints.unit_interval, 1),
        "logits": constraints.real_vector,
    }

    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args=None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("either probs or logits must be set, not both")
        if probs is not None:
            self.probs = self._param = probs
        else:
            self.logits = self._param = logits
        shape = self._param.shape
        if len(shape) < 1:
            raise ValueError("param must be at least 1 dimensional")
        super().__init__(shape[:-1], validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._param.size(-1))

    def enumerate_support(self, expand=True) -> torch.Tensor:
        total = self.event_shape[0]
        support = torch.arange(total + 1, device=self._param.device, dtype=torch.float)
        support = support.view((-1,) + (1,) * self.batch_shape)
        if expand:
            support = support.expand((-1,) + self.batch_shape)
        return support

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, True)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, True)

    @lazy_property
    def log_r(self):
        logits = self.logits
        batch_shape = self.batch_shape
        batch_dims = len(batch_shape)
        if not batch_dims:
            logits = logits.unsqueeze(0)
        else:
            logits = logits.flatten(end_dim=-2)
        # unfortunately, the sums can get pretty big. This leads to NaNs in the
        # backward pass. We go straight into log here
        log_r_ell = torch.zeros_like(logits)
        log_r = [log_r_ell[:, -1]]
        for _ in range(logits.size(1)):
            log_r_ell = (logits + log_r_ell).logcumsumexp(1)
            log_r.append(log_r_ell[:, -1])
            logits, log_r_ell = logits[:, 1:], log_r_ell[:, :-1]
        log_r = torch.stack(log_r, 1)
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
        probs = self.probs
        probs = probs.expand(shape + probs.shape[-1:])
        with torch.no_grad():
            sample = torch.bernoulli(probs).sum(-1)
        return sample

    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        shape = value.shape
        log_r = self.log_r
        log_r = log_r.log_softmax(-1).expand(shape + log_r.shape[-1:])
        return log_r.gather(-1, value.long().unsqueeze(-1)).squeeze(-1)


@torch.jit.script_if_tracing
@torch.no_grad()
def extended_conditional_bernoulli(
    w_f: torch.Tensor,
    given_count: torch.Tensor,
    r_f: Optional[torch.Tensor] = None,
    log_space: bool = False,
    neg_inf: float = config.EPS_NINF,
) -> torch.Tensor:
    device = w_f.device
    if w_f.dim() == 2:
        # the user passed regular odds instead of the relevant odds forward.
        N, T = w_f.size()
        # call checks given_count size
        w_f = extract_relevant_odds_forward(
            w_f, given_count, True, neg_inf if log_space else 0
        )
    else:
        _, N, diffmax = w_f.size()
        T = diffmax + int(given_count.min()) - 1
        assert (
            device == given_count.device
            and given_count.dim() == 1
            and given_count.size(0) == N
        )
    if r_f is None:
        r_f = (
            log_R_forward(w_f, given_count, True)
            if log_space
            else R_forward(w_f, given_count, True)
        )
    else:
        assert (
            r_f.dim() == 3
            and r_f.size(0) == w_f.size(0) + 1
            and r_f.size(1) == N
            and r_f.size(2) == diffmax
        )
    assert r_f.dim() == 3
    if log_space:
        return _log_extended_conditional_bernoulli(
            w_f, given_count.long(), r_f, T, neg_inf
        )
    else:
        return _extended_conditional_bernoulli(w_f, given_count.long(), r_f, T)


@torch.jit.script_if_tracing
@torch.no_grad()
def _extended_conditional_bernoulli(
    w_f: torch.Tensor, given_count: torch.Tensor, r_f: torch.Tensor, T: int
) -> torch.Tensor:
    device = w_f.device
    lmax_max, N, diffmax = w_f.size()
    remainder = given_count
    nrange = torch.arange(N)
    drange = torch.arange(diffmax)
    lmax_max = r_f.size(0) - 1
    tau_ellp1 = torch.full((N, 1), diffmax)
    b = torch.zeros((N, T), device=device, dtype=torch.float)
    for _ in range(lmax_max):
        valid_ell = remainder > 0
        mask_ell = (tau_ellp1 < drange) & valid_ell.unsqueeze(1)
        weights_ell = (
            w_f[remainder - 1, nrange] * r_f[remainder - 1, nrange]
        ).masked_fill(mask_ell, 0)
        tau_ell = torch.multinomial(weights_ell + (~valid_ell).unsqueeze(1), 1, True)
        remainder = (remainder - 1).clamp_min_(0)
        b[nrange, (tau_ell.squeeze(1) + remainder).clamp_max_(T - 1)] += valid_ell
        tau_ellp1 = tau_ell
    return b[:, :T]


@torch.jit.script_if_tracing
@torch.no_grad()
def _log_extended_conditional_bernoulli(
    logits_f: torch.Tensor,
    given_count: torch.Tensor,
    log_r_f: torch.Tensor,
    T: int,
    neg_inf: float = config.EPS_NINF,
) -> torch.Tensor:
    device = logits_f.device
    lmax_max, N, diffmax = logits_f.size()
    remainder = given_count
    nrange = torch.arange(N, device=device)
    drange = torch.arange(diffmax, device=device)
    lmax_max = log_r_f.size(0) - 1
    tau_ellp1 = torch.full((N, 1), diffmax, device=device)
    b = torch.zeros((N, T), device=device, dtype=torch.float)
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
        b[nrange, (tau_ell.squeeze(1) + remainder).clamp_max_(T - 1)] += valid_ell
        tau_ellp1 = tau_ell
    return b[:, :T]


class ConditionalBernoulli(torch.distributions.ExponentialFamily):

    arg_constraints = {
        "given_count": constraints.nonnegative_integer,
        "probs": constraints.independent(constraints.unit_interval, 1),
        "logits": constraints.real_vector,
    }
    _mean_carrier_measure = 0

    def __init__(
        self,
        given_count: Union[torch.Tensor, int],
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args=None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("either probs or logits must be set, not both")
        if probs is not None:
            self.probs = self._param = probs
        else:
            self.logits = self._param = logits.clamp_min(config.EPS_NINF)
        shape = self._param.shape
        if len(shape) < 1:
            raise ValueError("param must be at least 1-dimensional")
        batch_shape, event_shape = shape[:-1], shape[-1:]
        self.given_count = (
            torch.as_tensor(given_count).expand(batch_shape).type_as(self._param)
        )
        self.L = int(self.given_count.max().item())
        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def has_enumerate_support(self) -> bool:
        return (self.given_count == self.given_count.flatten()[0]).all().item()

    @constraints.dependent_property
    def support(self) -> torch.Tensor:
        return _d.BinaryCardinalityConstraint(self.given_count, self._event_shape[0],)

    def enumerate_support(self, expand=True) -> torch.Tensor:
        if not self.has_enumerate_support:
            raise NotImplementedError(
                "given_count must all be equal to enumerate support"
            )
        total = self._event_shape[0]
        given = int(self.given_count.flatten()[0].item())
        support = _f.enumerate_binary_sequences_with_cardinality(
            total, given, dtype=torch.float
        )
        support = support.view((-1,) + (1,) * len(self.batch_shape) + (total,))
        if expand:
            support = support.expand((-1,) + self.batch_shape + (total,))
        return support

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, True)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, True)

    @lazy_property
    def logits_f(self):
        logits, given_count = self.logits, self.given_count
        batch_dims = len(self.batch_shape)
        if not batch_dims:
            logits, given_count = logits.unsqueeze(0), given_count.view(1)
        else:
            logits = logits.flatten(end_dim=-2)
            given_count = given_count.flatten()
        logits_f = extract_relevant_odds_forward(logits, given_count, -float("inf"))
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
        log_r_f = log_R_forward(logits_f, given_count, True)
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
    def mean(self):
        # FIXME(sdrobert): there are faster ways of computing this.
        logits = self.logits
        batch_dims = len(self.batch_shape)
        if not batch_dims:
            logits = logits.flatten(end_dim=-2)
        N, T = logits.shape
        mask = torch.eye(T, device=logits.device, dtype=torch.bool).expand(N, T, T)
        x = logits.unsqueeze(1).expand(N, T, T)
        x = x.masked_fill(mask, config.EPS_NINF)
        x = x.view(N * T, T)
        given_count = self.given_count.flatten()
        empty = given_count == 0
        lm1 = (given_count - 1).clamp_min_(0).repeat_interleave(T)
        x = extract_relevant_odds_forward(x, lm1, config.EPS_NINF)
        x = log_R_forward(x, lm1, False).view(N, T)
        x = (x + logits).masked_fill(empty.unsqueeze(1), config.EPS_NINF)
        if batch_dims:
            x = x.unflatten(0, self.batch_shape)
        else:
            x = x.squeeze(0)
        return (x - self.log_partition.unsqueeze(-1)).exp()

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
        new.L = self.L
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size([])):
        sample_shape = torch.Size(sample_shape)
        shape = self._extended_shape(sample_shape)
        M = 1
        for s in sample_shape:
            M *= s
        if not M:
            return torch.empty(
                sample_shape, device=self._param.device, dtype=torch.long
            )
        logits_f, log_r_f, given_count = self.logits_f, self.log_r_f, self.given_count
        with torch.no_grad():
            if not len(self.batch_shape):
                logits_f = logits_f.unsqueeze(1).expand(-1, M, 1)
                log_r_f = log_r_f.unsqueeze(1).expand(-1, M, 1)
                given_count = given_count.expand(M)
            else:
                L, D = logits_f.size(0), logits_f.size(-1)
                assert D == log_r_f.size(-1) and L == self.L == log_r_f.size(0) - 1
                logits_f = logits_f.view(L, 1, -1, D).expand(L, M, -1, D).flatten(1, 2)
                log_r_f = (
                    log_r_f.view(L + 1, 1, -1, D).expand(L + 1, M, -1, D).flatten(1, 2)
                )
                given_count = given_count.view(1, -1).expand(M, -1).flatten()
            b = extended_conditional_bernoulli(logits_f, given_count, log_r_f, True)
        return b.view(shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        num = (
            (value * self.logits.expand_as(value))
            .sum(-1)
            .masked_fill(value.sum(-1) != self.given_count, config.EPS_NINF)
        )
        return num - self.log_partition

    def marginal_log_likelihoods(
        self, lcond: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        shape = self.batch_shape + self.event_shape
        if lcond.shape[1:] != shape or lcond.shape[0] < self.L:
            raise ValueError(
                f"Incorrect shape. Expecting (>={self.L}), + {tuple(shape)}, got "
                f"{tuple(lcond.shape)}"
            )
        logits_f, given_count = self.logits_f, self.given_count
        if not len(self.batch_shape):
            lcond = lcond.unsqueeze(1)
            logits_f = logits_f.unsqueeze(1)
            given_count = given_count.view(1)
        else:
            lcond = lcond.flatten(1, -2)
            logits_f = logits_f.flatten(1, -2)
            given_count = given_count.flatten()
        lcond_f = extract_relevant_odds_forward(lcond, given_count, config.EPS_NINF)
        assert lcond_f.shape == logits_f.shape
        ll = log_R_forward(logits_f + lcond_f, given_count)
        if normalize:
            ll = ll - self.log_partition
        return ll.view(self.batch_shape)


# TESTS


def test_poisson_binomial():
    # The Poisson Binomial equals the Binomial distribution when all probs are equal
    torch.manual_seed(1)
    T, N, M = 5, 16, 2 ** 15
    total_count = torch.randint(T + 1, (N,))
    p = torch.rand(N, requires_grad=True)

    p_ = p.unsqueeze(1).expand(N, T)
    p_ = p_.masked_fill(total_count.unsqueeze(1) <= torch.arange(T), 0)
    poisson_binom = PoissonBinomial(probs=p_)
    b = poisson_binom.sample([M])
    mean = b.mean(0)
    assert torch.allclose(mean, p * total_count, atol=1e-2)

    log_prob_act = poisson_binom.log_prob(b).t()
    (grad_log_prob_act,) = torch.autograd.grad(log_prob_act.mean(), [p])

    binom = torch.distributions.Binomial(total_count, probs=p)
    log_prob_exp = binom.log_prob(b).t()
    assert torch.allclose(log_prob_exp, log_prob_act, atol=1e-3)
    (grad_log_prob_exp,) = torch.autograd.grad(log_prob_exp.mean(), [p])
    assert torch.allclose(grad_log_prob_exp, grad_log_prob_act, atol=1e-3)


def test_conditional_bernoulli():
    torch.manual_seed(2)
    T, N, M = 10, 10, 2 ** 15
    logits = torch.randn(N, T, requires_grad=True)
    idx_1_first = []
    idx_1_second = []
    for first in range(T - 1):
        for second in range(first + 1, T):
            idx_1_first.append(first)
            idx_1_second.append(second)
    idx_1_first = torch.tensor(idx_1_first)
    idx_1_second = torch.tensor(idx_1_second)
    logits_ = logits[:, idx_1_first] + logits[:, idx_1_second]
    probs_ = logits_.softmax(1)  # (N, s)
    first_matches = torch.arange(T).unsqueeze(1) == idx_1_first  # (T, s)
    second_matches = torch.arange(T).unsqueeze(1) == idx_1_second  # (T, s)
    inclusions_exp = (probs_.unsqueeze(1) * (first_matches | second_matches)).sum(2)
    assert torch.allclose(inclusions_exp.sum(1), torch.tensor(2.0))
    conditional_bernoulli = ConditionalBernoulli(2, logits=logits)

    b = conditional_bernoulli.sample([M]).float()
    assert b.shape == torch.Size([M, N, T])
    assert (b.sum(-1) == 2).all()
    assert ((b == 0.0) | (b == 1.0)).all()
    inclusions_act = b.mean(0)
    assert torch.allclose(inclusions_exp, inclusions_act, atol=1e-2)

    poisson_binomial = PoissonBinomial(logits=logits)
    log_prob_act = conditional_bernoulli.log_prob(b) + poisson_binomial.log_prob(
        torch.tensor(2).expand((N,))
    )
    (grad_log_prob_act,) = torch.autograd.grad(log_prob_act.mean(), [logits])

    bernoulli = torch.distributions.Bernoulli(logits=logits)
    log_prob_exp = bernoulli.log_prob(b).sum(-1)
    assert log_prob_exp.shape == log_prob_act.shape
    assert torch.allclose(log_prob_exp, log_prob_act)
    (grad_log_prob_exp,) = torch.autograd.grad(log_prob_exp.mean(), [logits])
    assert torch.allclose(grad_log_prob_exp, grad_log_prob_act)

    given_count = torch.randint(T + 1, (N,))
    conditional_bernoulli = ConditionalBernoulli(
        given_count, logits=logits, validate_args=True
    )
    b = conditional_bernoulli.sample([M]).float()
    assert (b.sum(-1) == given_count.unsqueeze(0)).all()
    assert (b.max(-1)[0] <= 1).all()

    inclusions_act = conditional_bernoulli.mean
    assert torch.allclose(inclusions_act.sum(1), given_count.float())
    assert torch.allclose(b.mean(0), inclusions_act, atol=1e-2)
    for n in range(N):
        given_count_n = int(given_count[n].item())
        inclusions_act_n = inclusions_act[n]
        logits_n = logits[n]
        conditional_bernoulli_n = ConditionalBernoulli(
            given_count_n, logits=logits_n, validate_args=True
        )

        assert conditional_bernoulli_n.has_enumerate_support
        b = conditional_bernoulli_n.enumerate_support().float()
        lpb_exp = torch.distributions.Bernoulli(logits=logits_n).log_prob(b).sum(1)
        lpb_exp -= lpb_exp.logsumexp(0)
        lpb_act = conditional_bernoulli_n.log_prob(b)
        assert torch.allclose(lpb_exp, lpb_act, atol=1e-5)

        pb = lpb_act.exp()
        assert torch.allclose(pb.sum(), torch.tensor(1.0))
        inclusions_exp_n = (pb.unsqueeze(1) * b).sum(0)
        assert torch.allclose(inclusions_exp_n, inclusions_act_n)


# XXX(anon): The following did not work. I've had difficulty with algorithms based
# on subtraction/division.
# @torch.no_grad()
# def extended_conditional_bernoulli(
#     w_f: torch.Tensor,
#     L: torch.Tensor,
#     kmax: int = 1,
#     r_f: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     assert kmax == 1, "kmax > 1 not yet implemented"
#     device = w_f.device
#     if w_f.dim() == 2:
#         # the user passed regular odds instead of the relevant odds forward.
#         N, T = w_f.size()
#         # call checks L size
#         w_f = extract_relevant_odds_forward(w_f, L, True)
#     else:
#         _, N, diffmax = w_f.size()
#         T = diffmax + int(L.min()) - 1
#         assert device == L.device and L.dim() == 1 and L.size(0) == N
#     if r_f is None:
#         r_f = R_forward(w_f, L, kmax, True, True)
#     else:
#         assert (
#             r_f.dim() == 3
#             and r_f.size(0) == w_f.size(0) + 1
#             and r_f.size(1) == N
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
#     b = torch.empty((T, N), device=device)
#     idx_0 = L.long()  # a.k.a. how many samples left to draw, ell
#     idx_1 = torch.arange(N, device=device)  # batch dim
#     idx_2 = T - idx_0  # a.k.a. (prefix size - ell)
#     u_k = torch.rand((N,), device=device) * r_f[idx_0, idx_1, idx_2]
#     for t in range(T - 1, -1, -1):
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
