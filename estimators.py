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

import math

from typing import Optional, Callable

import torch
import pydrobert.torch.estimators as _e
import pydrobert.torch.distributions as _d
import pydrobert.torch.functional as _f

import distributions

ProposalMaker = Callable[[torch.Tensor, int], torch.distributions.Distribution]
# in: sufficient statistic and iteration #, out: adapted proposal


class FixedCardinalityGibbsStatistic(object):

    func: _e.FunctionOnSample
    density: _d.Density
    is_log: bool

    def __init__(
        self, func: _e.FunctionOnSample, density: _d.Density, is_log=False
    ) -> None:
        self.func = func
        self.density = density
        self.is_log = is_log

    @torch.no_grad()
    def __call__(self, b: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        T, D, device = b.size(-1), b.dim(), b.device
        reps = torch.eye(T, device=device)
        reps = reps.unsqueeze(0) - reps.unsqueeze(1)
        reps = reps.view((T ** 2,) + (1,) * (D - 1) + (T,))
        b_ = reps + b
        keep_mask = (b_ != 2).all(-1) & (b_ != -1).all(-1)
        keep_mask[T + 1 :: T + 1] = False  # only calculate the original values once
        b_ = b_[keep_mask]
        if D > 1:
            keep_mask_lens = keep_mask.sum(0)
            len_mask = torch.arange(keep_mask_lens.max(), device=device).view(
                (-1,) + (1,) * keep_mask_lens.dim()
            ) < keep_mask_lens.unsqueeze(0)
            b_ = torch.zeros(
                len_mask.shape + (T,), device=device, dtype=b_.dtype
            ).masked_scatter_(len_mask.unsqueeze(-1).expand(len_mask.shape + (T,)), b_)
        ll_ = torch.zeros_like(b_[..., 0])
        for idx in range(0, b_.size(0), chunk_size):
            b_i = b_[idx : idx + chunk_size]
            ll_i = self.func(b_i)
            if not self.is_log:
                ll_i = ll_i.abs_().log_()
            ll_i += self.density.log_prob(b_i)
            ll_[idx : idx + chunk_size] = ll_i
        if D > 1:
            ll_ = ll_[len_mask]
        ll = torch.full(keep_mask.shape, -float("inf"), device=device)
        ll.masked_scatter_(keep_mask, ll_)
        ll = ll.movedim(0, -1)
        # the likelihood of the original value only applies when the binary value
        # was high in the original sample
        ll[..., :: T + 1] = ll[..., :1].masked_fill(b == 0, -float("inf"))
        h = ll.unflatten(-1, (T, T)).softmax(-1).nan_to_num_(0).sum(-2)
        return h


class ConditionalBernoulliProposalMaker(object):

    eps: float

    def __init__(
        self, given_count: torch.Tensor, eps: float = torch.finfo(torch.float).eps
    ):
        self.eps = eps
        self.given_count = given_count

    def __call__(self, h_n: torch.Tensor, n: int) -> distributions.ConditionalBernoulli:
        return distributions.ConditionalBernoulli(
            self.given_count, h_n.clamp(self.eps, 1 - self.eps)
        )


class AisImhEstimator(_e.IndependentMetropolisHastingsEstimator):

    adaptation_func: _e.FunctionOnSample
    proposal_maker: ProposalMaker
    density: torch.distributions.Distribution

    def __init__(
        self,
        proposal: torch.distributions.Distribution,
        func: _e.FunctionOnSample,
        mc_samples: int,
        density: torch.distributions.Distribution,
        adaptation_func: _e.FunctionOnSample,
        proposal_maker: ProposalMaker,
        burn_in: int = 1,
        initial_sample: Optional[torch.Tensor] = None,
        initial_sample_tries: int = 1000,
        is_log: bool = False,
    ) -> None:
        self.adaptation_func = adaptation_func
        self.proposal_maker = proposal_maker
        if burn_in < 1:
            raise ValueError("burn_in must be at least 1")
        super().__init__(
            proposal,
            func,
            mc_samples,
            density,
            burn_in,
            initial_sample,
            initial_sample_tries,
            is_log,
        )

    def __call__(self) -> torch.Tensor:
        if self.initial_sample is None:
            last_sample = self.find_initial_sample()
        else:
            last_sample = self.initial_sample
        laf = self.func(last_sample)  # log(abs(func(b)))
        if not self.is_log:
            laf = laf.abs().log()
        last_density = laf + self.density.log_prob(last_sample)
        if self.is_log:
            lomegas = []
        else:
            omegas = []
        h_last = self.adaptation_func(last_sample.squeeze(0))
        h_last_last = None
        # a = 1.0
        for n in range(1, self.mc_samples + 1):
            if n <= self.burn_in:
                Q = self.proposal
            else:
                Q = self.proposal_maker(h_last_last, n - 2)
            sample = Q.sample([1])
            laf = self.func(sample)
            lpd = self.density.log_prob(sample)
            with torch.no_grad():
                lpp = Q.log_prob(sample)
            if self.is_log:
                density = laf + lpd
                lomegas.append(density - lpp)
            else:
                omegas.append(laf * (lpd - lpp).exp())
                laf = laf.abs().log()
                density = laf + lpd
            with torch.no_grad():
                lpp_last = Q.log_prob(last_sample)
            accept_ratio = (density - last_density - lpp + lpp_last).exp()
            u = torch.rand_like(accept_ratio)
            accept = u < accept_ratio
            # a = a * ((n - 1) / n) + accept.float().mean() / n
            last_density = torch.where(accept, density, last_density)
            accept = accept.view(accept.shape + (1,) * len(Q.event_shape))
            tau = torch.where(accept, sample, last_sample)
            h_last_last = h_last
            h_last = (
                h_last_last * ((n - 1) / n) + self.adaptation_func(tau.squeeze(0)) / n
            )
            # if n % 1024 == 0:
            #     print(n, h_last, a)
        # cat not stack b/c 0 dim is sample dim
        if self.is_log:
            v = torch.cat(lomegas).logsumexp(0) - math.log(self.mc_samples)
        else:
            v = torch.cat(omegas).mean(0)
        return v


class SerialMCWrapper(_e.MonteCarloEstimator):

    wrapped: _e.MonteCarloEstimator
    chunk_size: int

    def __init__(self, wrapped: _e.MonteCarloEstimator, chunk_size: int = 1) -> None:
        super().__init__(
            wrapped.proposal, wrapped.func, wrapped.mc_samples, wrapped.is_log
        )
        self.chunk_size = chunk_size
        self.wrapped = wrapped

    def __call__(self) -> torch.Tensor:
        try:
            v = None
            remainder = self.mc_samples
            while remainder:
                cur_chunk_size = min(remainder, self.chunk_size)
                self.wrapped.mc_samples = cur_chunk_size
                v_cur = self.wrapped() * cur_chunk_size
                if v is None:
                    v = v_cur
                elif self.is_log:
                    v = v + (v_cur - v).exp().log1p()
                else:
                    v = v + v_cur
                remainder -= cur_chunk_size
                del v_cur
            v = v / self.mc_samples
        finally:
            self.wrapped.mc_samples = self.mc_samples
        return v


## ==== TESTS


class DummyLogFunc(torch.nn.Module):

    L: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor

    def __init__(self, y: torch.Tensor, x: torch.Tensor, L: torch.Tensor) -> None:
        super().__init__()
        self.L = L
        self.y, self.x = y, x
        assert x.shape == y.shape
        assert L.dim() == 1 and L.size(0) == y.size(0)
        self.weights = torch.nn.parameter.Parameter(torch.empty(y.size(1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_()

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        assert b.shape[1:] == self.y.shape
        y_act = self.x * self.weights.unsqueeze(0)  # (N, T)
        sse = (self.y - y_act) ** 2
        return -(b * sse).sum(2) - 0.01 * (b.sum(2) - self.L) ** 2


class DummyBernoulliSequence(torch.distributions.distribution.Distribution):

    has_enumerate_support = True

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        super().__init__(logits.shape[:-1], logits.shape[1:], validate_args=False)

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.logits.sigmoid().expand(shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        lp = -torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits.expand_as(value), value.float(), reduction="none"
        )
        return lp.sum(-1)

    def enumerate_support(self, expand=True):
        support = _f.enumerate_binary_sequences(self.event_shape[0], self.logits.device)
        support = support.view((-1,) + (1,) * len(self.batch_shape) + self.event_shape)
        if expand:
            support = support.expand((-1,) + self.batch_shape + self.event_shape)
        return support


class DummyGibbs(object):

    log_func: DummyLogFunc
    density: DummyBernoulliSequence

    def __init__(self, log_func, density) -> None:
        self.log_func = log_func
        self.density = density

    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        T = self.density.event_shape[0]
        b_ = b.unsqueeze(0)
        ones_mask = torch.eye(T, device=b.device).view(
            (T,) + (1,) * len(self.density.batch_shape) + (T,)
        )
        pos_instances = (b_ + ones_mask).clamp_max_(1)
        neg_instances = (b_ - ones_mask).clamp_min_(0)
        pos_likelihoods = self.log_func(pos_instances) + self.density.log_prob(
            pos_instances
        )
        neg_likelihoods = self.log_func(neg_instances) + self.density.log_prob(
            neg_instances
        )
        return (pos_likelihoods - neg_likelihoods).movedim(0, -1).sigmoid()


def _test_estimator_boilerplate(T, N):
    logits = torch.randn(N, T, requires_grad=True)
    y = torch.randn(N, T)
    x = torch.randn(N, T)
    L = torch.randint(T + 1, (N,))
    func = DummyLogFunc(y, x, L)
    dist = DummyBernoulliSequence(logits)
    estimator = _e.EnumerateEstimator(dist, func, is_log=True)
    z_exp = estimator()
    g_exp_logits, g_exp_theta = torch.autograd.grad(
        z_exp, [logits, func.weights], torch.ones_like(z_exp)
    )
    return logits, func, z_exp, g_exp_logits, g_exp_theta


def test_dummy():
    torch.manual_seed(1)
    T, N, M = 5, 10, 2 ** 13
    logits, func, z_exp, g_exp_logits, g_exp_theta = _test_estimator_boilerplate(T, N)
    dist = DummyBernoulliSequence(logits)
    estimator = _e.DirectEstimator(dist, func, M, is_log=True)
    z_act = estimator()
    g_act_logits, g_act_theta = torch.autograd.grad(
        z_act, [logits, func.weights], torch.ones_like(z_act)
    )
    assert torch.allclose(z_exp, z_act, atol=1e-1)
    assert torch.allclose(g_exp_logits, g_act_logits, atol=1e-1)
    assert torch.allclose(g_exp_theta, g_act_theta, atol=1e-1)


def test_ais_imh():
    torch.manual_seed(1)
    T, N, M = 5, 10, 2 ** 9
    logits, func, z_exp, g_exp_logits, g_exp_theta = _test_estimator_boilerplate(T, N)
    density = DummyBernoulliSequence(logits)
    proposal = DummyBernoulliSequence(torch.zeros_like(logits))
    adapt = DummyGibbs(func, density)
    estimator = AisImhEstimator(
        proposal,
        func,
        M,
        density,
        adapt,
        lambda h, n: DummyBernoulliSequence(
            torch.logit(h, torch.finfo(torch.float).tiny)
        ),
        M // 4,
        is_log=True,
    )
    z_act = estimator()
    g_act_logits, g_act_theta = torch.autograd.grad(
        z_act, [logits, func.weights], torch.ones_like(z_act)
    )
    assert torch.allclose(z_exp, z_act, atol=1e-1), (z_exp - z_act).abs().max().item()
    assert torch.allclose(g_exp_logits, g_act_logits, atol=1e-1), (
        (g_exp_logits - g_act_logits).abs().max().item()
    )
    assert torch.allclose(g_exp_theta, g_act_theta, atol=1e-1)


def test_fixed_cardinality_gibbs_statistic():
    torch.manual_seed(2)
    N, T = 5, 10
    L = torch.randint(T + 1, (N,))
    logits = torch.randn(N, T)
    func = lambda x: torch.zeros(x.shape[:-1])
    density = distributions.ConditionalBernoulli(L, logits=logits, validate_args=False)
    gibbs = FixedCardinalityGibbsStatistic(func, density, True)
    b = _f.enumerate_binary_sequences(T).unsqueeze(1).expand(-1, N, T)
    probs = density.log_prob(b).exp()
    assert torch.allclose(probs.sum(0), torch.tensor(1.0))
    act = (gibbs(b) * probs.unsqueeze(-1)).sum(0)
    assert torch.allclose(act, density.mean)
