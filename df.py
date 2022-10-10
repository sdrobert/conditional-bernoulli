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

"""Sequential classification based on Drezner & Farnum 1993"""

import math

from typing import Optional, Dict, Tuple

import torch
from torch.distributions.utils import logits_to_probs, clamp_probs

from pydrobert.torch.distributions import SequentialLanguageModelDistribution
from pydrobert.torch.modules import MixableSequentialLanguageModel, RandomWalk

torch.use_deterministic_algorithms(True)
torch.set_printoptions(precision=2, sci_mode=False)


class DreznerFarnumBernoulli(MixableSequentialLanguageModel):
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
    init_theta_1
        The initial value for :math:`\theta_1`. If unset, :math:`\theta_1 \sim
        \mathcal{N}(0, 1)`. Setting to infinity disables that parameter's gradient.
    init_theta_2
        The initial value for :math:`\theta_2`. If unset, :math:`\theta_2 \sim
        \mathcal{N}(0, 1)`. Setting to infinity disables that parameter's gradient.
    
    Variables
    ---------
    theta_1
        The 1-d variable for :math:`\theta_1`.
    theta_2
        The 1-d variable for :math:`\theta_2`.
    """

    __constants__ = ["init_theta_1", "init_theta_2"]

    init_theta_1: Optional[float]
    init_theta_2: Optional[float]

    def __init__(
        self, init_theta_1: Optional[float] = None, init_theta_2: Optional[float] = None
    ):
        super().__init__(2)
        self.init_theta_1 = init_theta_1
        self.init_theta_2 = init_theta_2
        self.theta_1 = torch.nn.Parameter(torch.empty(1))
        self.theta_2 = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_theta_1 is not None:
            self.theta_1.data.fill_(self.init_theta_1)
            if math.isinf(self.init_theta_1):
                self.theta_1.requires_grad_(False)
        else:
            self.theta_1.data.normal_()
        if self.init_theta_2 is not None:
            self.theta_2.data.fill_(self.init_theta_2)
            if math.isinf(self.init_theta_2):
                self.theta_2.requires_grad_(False)
        else:
            self.theta_2.data.normal_()

    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return prev  # no state

    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return prev_true  # no state

    @property
    def prob_1(self) -> torch.Tensor:
        return logits_to_probs(self.theta_1, True)

    @property
    def prob_2(self) -> torch.Tensor:
        return logits_to_probs(self.theta_2, True)

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        T, N = hist.shape
        p1, p2 = self.prob_1, self.prob_2
        lp_0 = torch.cat([(1 - p1).log(), p1.log()])
        lpind = lp_0 + (1 - p2).log()
        lp_0, lpind = lp_0.expand(N, 2), lpind.expand(N, 2)
        if T == 0:
            return lp_0, prev
        denom = idx.clamp_min(1)
        idx = idx.expand(N)
        mean = (
            hist.cumsum(0).gather(0, (idx - 1).clamp_min_(0).unsqueeze(0)).squeeze(0)
            / denom
        )
        mean = clamp_probs(mean)
        lpdep = torch.stack([(p2 * (1 - mean)).log(), (p2 * mean).log()], 1)
        max_, min_ = torch.max(lpind, lpdep), torch.min(lpind, lpdep)
        lp = max_ + (min_ - max_).exp().log1p()
        lp = torch.where((idx > 0).unsqueeze(1), lp, lp_0)
        return lp, prev

    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        T, N = hist.shape
        if T < 1:
            raise RuntimeError("history is empty")
        p1, p2 = self.prob_1, self.prob_2
        device, dtype = p1.device, p1.dtype
        lp_0 = torch.cat([(1 - p1).log(), p1.log()])
        lpind = lp_0 + (1 - p2).log()
        lp_0, lpind = lp_0.expand(1, N, 2), lpind.expand(T, N, 2)
        mean = hist.cumsum(0, dtype=dtype) / torch.arange(
            1, T + 1, device=device, dtype=dtype
        ).unsqueeze(1)
        mean = clamp_probs(mean)
        lpdep = torch.stack([(p2 * (1 - mean)).log(), (p2 * mean).log()], 2)
        max_, min_ = torch.max(lpind, lpdep), torch.min(lpind, lpdep)
        lp = max_ + (min_ - max_).exp().log1p()
        lp = torch.cat([lp_0, lp])
        return lp


class GaussianConditional(torch.nn.Module):
    def __init__(self, in_size: int) -> None:
        super().__init__()
        self.W = torch.nn.Linear(in_size, 2, bias=False)

    def reset_parameters(self):
        self.W.reset_parameters()

    def get_stats(
        self, x: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.masked_select(b.unsqueeze(2).bool()).view(-1, x.size(2))
        x = self.W(x)
        return x[..., 0], x[..., 1].sigmoid()

    def forward(
        self, y: torch.nn.utils.rnn.PackedSequence, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        T, N = x.shape[:-1]
        assert b.shape == (T, N)
        if y.sorted_indices is not None:
            x = x.index_select(1, y.sorted_indices)
            b = b.index_select(1, y.sorted_indices)
        lens_y = (torch.arange(N).unsqueeze(1) < y.batch_sizes).sum(1)
        lens_b = b.T.long().sum(1).cpu()
        b = torch.where(
            lens_y == lens_b,
            b.bool(),
            (torch.arange(T).unsqueeze(1) < lens_y).to(b.device),
        )
        mu, std = self.get_stats(x, b)
        x = torch.nn.functional.gaussian_nll_loss(
            y.data, mu, std, full=True, reduction="none"
        )
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(x, *y[1:]), batch_first=True
        ).sum(1)
        x = x.masked_fill(lens_y != lens_b, -10_000)
        return x

    @torch.no_grad()
    def sample(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mu, std = self.get_stats(x, b)
        y = torch.distributions.Normal(mu, std).sample()


def test_drezner_farnum_bernoulli():
    torch.manual_seed(1)
    T, N, M = 5, 10, 2 ** 15
    df_bernoulli = DreznerFarnumBernoulli()
    zero = torch.tensor(0.0)
    one = 1 - zero

    # first check if calc_full_log_probs yields the same results as a manual roll out
    hist = torch.randint(2, (T, N))
    log_probs_exp = df_bernoulli(hist)
    assert log_probs_exp.shape == (T + 1, N, 2)
    assert torch.allclose(log_probs_exp.exp().sum(-1), one)
    g_th1_exp, g_th2_exp = torch.autograd.grad(
        log_probs_exp[..., 0],
        df_bernoulli.parameters(),
        torch.ones_like(log_probs_exp[..., 0]),
    )
    log_probs_act = []
    for t in range(T + 1):
        log_probs_act.append(df_bernoulli(hist, idx=t)[0])
    log_probs_act = torch.stack(log_probs_act)
    assert log_probs_exp.shape == log_probs_act.shape
    assert torch.allclose(log_probs_exp, log_probs_act)
    g_th1_act, g_th2_act = torch.autograd.grad(
        log_probs_act[..., 0],
        df_bernoulli.parameters(),
        torch.ones_like(log_probs_act[..., 0]),
    )
    assert torch.isclose(g_th1_exp, g_th1_act)
    assert torch.isclose(g_th2_exp, g_th2_act)

    # now plug into the distribution
    p, omp = df_bernoulli.prob_1, 1 - df_bernoulli.prob_1
    walk = RandomWalk(df_bernoulli)
    dist = SequentialLanguageModelDistribution(walk, max_iters=T)
    sample = dist.sample([M])
    assert torch.allclose(sample.float().mean(0), p, atol=1e-2)
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean() / T
    assert torch.isfinite(Elp_act)
    g_th1, g_th2 = torch.autograd.grad(Elp_act, walk.parameters())
    assert torch.isclose(g_th1, zero, atol=1e-2)
    assert torch.isclose(g_th2, zero, atol=1e-2)

    # make the trials independent
    df_bernoulli.theta_2.data.fill_(-float("inf"))
    Elp_exp = p.detach() * p.log() + omp.detach() * omp.log()
    (g_th1,) = torch.autograd.grad(Elp_exp, df_bernoulli.theta_1)
    assert torch.isclose(g_th1, zero, atol=1e-2)
    sample = dist.sample([M])
    assert torch.allclose(sample.float().mean(0), p, atol=1e-2)
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean() / T
    assert torch.isclose(Elp_exp, Elp_act, atol=1e-2)
    (g_th1,) = torch.autograd.grad(Elp_act, df_bernoulli.theta_1)
    assert torch.isclose(g_th1, zero, atol=1e-2)

    # now make them entirely dependent
    df_bernoulli.theta_2.data.fill_(float("inf"))
    sample = dist.sample([M])
    assert torch.allclose(sample.float().mean(0), p, atol=1e-2)
    lp_act = dist.log_prob(sample)
    Elp_act = lp_act.mean()
    assert torch.isclose(Elp_exp, Elp_act, atol=1e-2)
    (g_th1,) = torch.autograd.grad(Elp_act, df_bernoulli.theta_1)
    assert torch.isclose(g_th1, zero, atol=1e-2)


# import config
# from estimators import (
#     CbAisImhEstimator,
#     ForcedSuffixIsEstimator,
#     ExtendedConditionalBernoulliEstimator,
#     EnumerateEstimator,
#     Estimator,
#     GibbsCbAisImhEstimator,
#     RejectionEstimator,
#     StaticSrsworIsEstimator,
#     Theta,
# )
# from typing import List, Tuple
# import param
# import argparse
# import sys
# import pydrobert.param.argparse as pargparse
# from pydrobert.param.serialization import DefaultObjectSelectorSerializer
# from tqdm import tqdm
# import pandas as pd
# import numpy as np
# from datetime import datetime


# @torch.jit.script
# @torch.no_grad()
# def dependent_bernoulli_sample(
#     theta_1: torch.Tensor, theta_2: torch.Tensor, tmax: int
# ) -> torch.Tensor:
#     r"""Sample from Drezner & Farnum's Dependent Bernoulli model

#     For a given batch element (and :math:`t` is 1-indexed)

#     .. math::

#         P(b_t = 1|b_{1:t-1}) = \sigma(\theta_1) * (1 - \sigma(\theta_2))
#             + \frac{\sigma(\theta_2)\sum_{t'=1}^{t-1}b_t}{t-1} \\
#         \sigma(\theta) = \frac{\exp(\theta)}{1 + \exp(\theta)}

#     Parameters
#     ----------
#     theta_1 : torch.Tensor
#         Shape ``(nmax,)`` of batch's independent logit, :math:`\theta_1`
#     theta_2 : torch.Tensor
#         Shape ``(nmax,)`` of batch's sample average mixing coefficient logit,
#         :math:`\theta_2`
#     tmax : int
#         Number of sequential Bernoulli values to sample

#     Returns
#     -------
#     b : torch.Tensor
#         Shape ``(tmax, nmax)`` where ``b[:, n]`` is the sequence of `tmax` sampled
#         Bernoulli values with parameters ``p[n]`` and ``beta[n]``.
#     """
#     assert tmax >= 0
#     assert theta_1.dim() == 1
#     nmax = theta_1.size(0)
#     assert theta_2.dim() == 1 and theta_2.size(0) == nmax
#     device = theta_1.device
#     assert theta_1.device == device
#     b = torch.empty((tmax, nmax), device=device, dtype=theta_1.dtype)
#     p_1, p_2 = theta_1.sigmoid(), theta_2.sigmoid()
#     if not tmax:
#         return b
#     b[0] = S_tm1 = torch.bernoulli(theta_1.sigmoid())
#     for tm1 in range(1, tmax):
#         p_t = (1.0 - p_2) * p_1 + p_2 * S_tm1 / tm1
#         b_t = torch.bernoulli(p_t)
#         b[tm1] = b_t  # 0-indexed
#         S_tm1 += b_t
#     return b


# @torch.jit.script
# def dependent_bernoulli_logprob(
#     theta_1: torch.Tensor,
#     theta_2: torch.Tensor,
#     b: torch.Tensor,
#     full: bool = False,
#     neg_inf: float = config.EPS_INF,
# ) -> torch.Tensor:
#     """Return the log-probability of sample under D & F Dependent Bernoulli model

#     Counterpart to :func:`dependent_bernoulli_sample`

#     Returns
#     -------
#     logprob : torch.Tensor
#         A tensor of shape ``(nmax,)`` where ``logprob[n]`` is the joint log-probability
#         of the Bernoulli sequence ``b[:, n]`` under the Drezner & Farnum model.
#     """
#     assert b.dim() == 2
#     assert theta_1.size() == theta_2.size() == b.size()[1:]
#     tmax, device = b.size(0), b.device
#     assert theta_1.device == theta_2.device == device
#     if not tmax:
#         return torch.empty((0, theta_1.size(0)), device=device, dtype=theta_1.dtype)
#     b = b.detach()
#     logp_1, log1mp_1 = theta_1.sigmoid().log(), (-theta_1).sigmoid().log()
#     logp_1_1 = (logp_1 * b[0]).nan_to_num_(neginf=neg_inf) + (
#         log1mp_1 * (1.0 - b[0])
#     ).nan_to_num_(neginf=neg_inf)
#     if tmax == 1:
#         return logp_1_1
#     S_tm1 = b[:-1].cumsum(0)
#     t_tm1 = torch.arange(1, tmax, device=device).unsqueeze(1)
#     logt_tm1 = t_tm1.log()
#     logpp_t, log1mpp_t = S_tm1.log() - logt_tm1, (t_tm1 - S_tm1).log() - logt_tm1
#     logp_2, log1mp_2 = theta_2.sigmoid().log(), (-theta_2).sigmoid().log()
#     b, negb = b[1:], 1.0 - b[1:]
#     left_t = (
#         (logp_1.unsqueeze(0) * b).nan_to_num_(neginf=neg_inf)
#         + (log1mp_1.unsqueeze(0) * negb).nan_to_num_(neginf=neg_inf)
#         + log1mp_2
#     )
#     right_t = (
#         (logpp_t * b).nan_to_num_(neginf=neg_inf)
#         + (log1mpp_t * negb).nan_to_num_(neginf=neg_inf)
#         + logp_2
#     )
#     max_t = torch.max(left_t, right_t).nan_to_num_()
#     logp_t = ((left_t - max_t).exp() + (right_t - max_t).exp()).log() + max_t
#     if full:
#         return torch.cat([logp_1_1.unsqueeze(0), logp_t])
#     else:
#         return logp_1_1 + logp_t.sum(0)


# @torch.jit.script
# @torch.no_grad()
# def event_sample(
#     theta_3: torch.Tensor, x: torch.Tensor, b: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     assert b.dim() == theta_3.dim() == 2
#     assert x.dim() == 3
#     assert x.size()[:2] == b.size()
#     isize = x.size(2)
#     assert isize == theta_3.size(0)
#     assert b.device == theta_3.device == x.device
#     lens = b.to(torch.long).sum(0)
#     x_at_b = (
#         x.transpose(0, 1)
#         .masked_select(b.t().to(torch.bool).unsqueeze(2))
#         .view(-1, isize)
#     )
#     logits = x_at_b @ theta_3
#     probs = logits.softmax(1)
#     y = torch.multinomial(probs, 1, True).flatten()
#     return y, lens


# @torch.jit.script
# def event_logprob(
#     theta_3: torch.Tensor, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor
# ) -> torch.Tensor:
#     assert b.dim() == theta_3.dim() == 2
#     assert x.dim() == 3
#     assert y.dim() == 1
#     assert x.size()[:2] == b.size()
#     isize = x.size(2)
#     assert isize == theta_3.size(0)
#     assert b.device == theta_3.device == x.device == y.device
#     b_ = b.t().to(torch.bool)
#     x_at_b = x.transpose(0, 1).masked_select(b_.unsqueeze(2)).view(-1, isize)
#     assert y.numel() == x_at_b.size(0), (
#         y.numel(),
#         x_at_b.size(),
#         b.size(),
#         b.sum(),
#         b[:, 0],
#         b_[0],
#     )
#     logits = x_at_b @ theta_3
#     logprob = logits.log_softmax(1).gather(1, y.unsqueeze(1)).squeeze(1)
#     return b.t().masked_scatter(b_, logprob).sum(1)


# @torch.jit.script
# def enumerate_latent_log_likelihoods(
#     theta_3: torch.Tensor,
#     x: torch.Tensor,
#     y: torch.Tensor,
#     lens: torch.Tensor,
#     neg_inf: float = config.EPS_INF,
# ) -> torch.Tensor:
#     assert theta_3.dim() == 2
#     assert x.dim() == 3
#     assert y.dim() == lens.dim() == 1
#     device = theta_3.device
#     assert device == x.device == y.device == lens.device
#     numel = torch.tensor(y.numel(), device=device)
#     assert numel == lens.sum()
#     limits = lens.cumsum(0)
#     offsets = torch.cat([torch.zeros(1, dtype=torch.long), limits[:-1]])
#     tmax, nmax, fmax = x.size()
#     lens_min, lens_max = int(lens.min().long().item()), int(lens.max().item())
#     x = torch.cat(
#         [x, torch.zeros((tmax - lens_min + lens_max + 1, nmax, fmax), device=device)]
#     )
#     if lens_max == 0:
#         return torch.zeros((0, tmax - lens_min + 1, nmax), device=device, dtype=x.dtype)
#     probs = (x @ theta_3).log_softmax(2)
#     g = []
#     for ell in range(lens_max):
#         y_ell = y[(offsets + ell).clamp_max_(numel - 1)]
#         probs_ell = probs[ell : tmax - lens_min + ell + 1]
#         g.append(
#             probs_ell.gather(
#                 2, y_ell.expand(probs_ell.size()[:-1]).unsqueeze(2)
#             ).squeeze(2)
#         )
#     mask = (
#         torch.arange(tmax - lens_min + 1, device=device).unsqueeze(1) >= tmax - lens + 1
#     )
#     mask = (torch.arange(lens_max).unsqueeze(1) >= lens).unsqueeze(1) | mask
#     return torch.stack(g, 0).masked_fill(mask, neg_inf)


# # For command-line task


# class DreznerFarnumBernoulliExperimentParameters(param.Parameterized):
#     seed = param.Integer(
#         None,
#         bounds=(-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF),
#         inclusive_bounds=(True, True),
#     )
#     num_trials = param.Integer(2 ** 10, bounds=(1, None))
#     train_batch_size = param.Integer(2 ** 6, bounds=(1, None))
#     kl_batch_size = param.Integer(2 ** 9, bounds=(1, None))
#     tmax = param.Integer(2 ** 5, bounds=(1, None))
#     fmax = param.Integer(2 ** 4, bounds=(1, None))
#     vmax = param.Integer(2 ** 4, bounds=(1, None))
#     p_1 = param.Magnitude(0.5)
#     p_2 = param.Magnitude(0.5)
#     x_std = param.Number(1.0, bounds=(0, None))
#     theta_3_std = param.Number(1.0, bounds=(0, None))
#     learning_rate = param.Magnitude(1e-1)
#     reduce_lr_patience = param.Integer(2 ** 4, bounds=(0, None))
#     reduce_lr_threshold = param.Number(1, bounds=(0, None))
#     reduce_lr_factor = param.Magnitude(1e-1, inclusive_bounds=(True, False))
#     reduce_lr_min = param.Magnitude(1e-2)
#     ais_burn_in = param.Integer(2 ** 5, bounds=(0, None))
#     estimator = param.ObjectSelector(
#         "srswor",
#         objects=(
#             "rej",  # Rejection sampling.
#             "fs",  # FS. Forced Suffix.
#             "srswor",  # SRSWOR. Simple Random Sampling WithOut Replacement.
#             "ecb",  # ECB. Extended Conditional Bernoulli.
#             "ais-cb-count",  # AIS-IMH. Count-based inclusion estimates.
#             "ais-cb-gibbs",  # AIS-IMH. Gibbs-style inclusion estimates.
#             "enum",  # Enumerate the support. Testing purposes only.
#         ),
#     )
#     optimizer = param.ObjectSelector(
#         torch.optim.Adam, objects={"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
#     )
#     num_mc_samples = param.Integer(2 ** 10, bounds=(1, None))


# def initialize(
#     df_params: DreznerFarnumBernoulliExperimentParameters,
# ) -> Tuple[
#     torch.optim.Optimizer,
#     torch.optim.lr_scheduler.ReduceLROnPlateau,
#     Theta,
#     Theta,
#     Estimator,
# ]:
#     dtype = torch.float
#     if df_params.seed is None:
#         df_params.seed = torch.randint(
#             torch.iinfo(torch.long).min, torch.iinfo(torch.long).max, (1,)
#         ).item()
#     torch.manual_seed(df_params.seed)
#     theta_1_exp = torch.tensor(df_params.p_1, dtype=dtype).logit_()
#     theta_2_exp = torch.tensor(df_params.p_2, dtype=dtype).logit_()
#     theta_3_exp = (
#         torch.randn((df_params.fmax, df_params.vmax), dtype=dtype)
#         * df_params.theta_3_std
#     )
#     theta_exp = [theta_1_exp, theta_2_exp, theta_3_exp]
#     theta_1_act = torch.randn(1, dtype=dtype).requires_grad_(True)
#     theta_2_act = torch.randn(1, dtype=dtype).requires_grad_(True)
#     theta_3_act = torch.randn(
#         (df_params.fmax, df_params.vmax), dtype=dtype, requires_grad=True
#     )
#     theta_act = [theta_1_act, theta_2_act, theta_3_act]
#     optim_params = theta_act.copy()
#     if df_params.p_2 == 0.0 or df_params.estimator == "ecb":
#         theta_2_act.data[0] = -float("inf")
#         optim_params.pop(1)
#     if df_params.estimator == "rej":
#         estimator = RejectionEstimator(
#             _psampler, df_params.num_mc_samples, _lp, _lg, theta_act
#         )
#     elif df_params.estimator == "fs":
#         estimator = ForcedSuffixIsEstimator(
#             _psampler, df_params.num_mc_samples, _lp_full, _lg, theta_act
#         )
#     elif df_params.estimator == "enum":
#         estimator = EnumerateEstimator(_lp, _lg, theta_act)
#     elif df_params.estimator == "srswor":
#         estimator = StaticSrsworIsEstimator(
#             df_params.num_mc_samples, _lp, _lg, theta_act
#         )
#     elif df_params.estimator == "ecb":
#         estimator = ExtendedConditionalBernoulliEstimator(
#             _ilw, _ilg, _lp, _lg, theta_act
#         )
#     elif df_params.estimator == "ais-cb-count":
#         estimator = CbAisImhEstimator(
#             _lie_count,
#             df_params.num_mc_samples,
#             _lp,
#             _lg,
#             theta_act,
#             df_params.ais_burn_in,
#         )
#     elif df_params.estimator == "ais-cb-gibbs":
#         estimator = GibbsCbAisImhEstimator(
#             df_params.num_mc_samples, _lp, _lg, theta_act, df_params.ais_burn_in
#         )
#     else:
#         raise NotImplementedError
#     optimizer = df_params.optimizer(optim_params, lr=df_params.learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=df_params.reduce_lr_factor,
#         patience=df_params.reduce_lr_patience,
#         threshold=df_params.reduce_lr_threshold,
#         min_lr=df_params.reduce_lr_min,
#     )
#     return optimizer, scheduler, theta_exp, theta_act, estimator


# def train_for_trial(
#     optimizer: torch.optim.Optimizer,
#     theta_exp: Theta,
#     estimator: Estimator,
#     df_params: DreznerFarnumBernoulliExperimentParameters,
# ) -> Tuple[float, float]:
#     optimizer.zero_grad()
#     x = (
#         torch.randn(
#             (df_params.tmax, df_params.train_batch_size, df_params.fmax),
#             dtype=theta_exp[0].dtype,
#         )
#         * df_params.x_std
#     )
#     b = _psampler(x, theta_exp)
#     # print(b[:10, 0])
#     events, lmax = event_sample(theta_exp[2], x, b)
#     y = _pad_events(events, lmax)
#     zhat, back, log_ess = estimator(x, y, lmax)
#     (-back).backward()
#     # grads_theta = torch.autograd.grad(-back, estimator.theta, allow_unused=True)
#     # for v, g in zip(estimator.theta, grads_theta):
#     #     if g is not None:
#     #         v.backward(g.nan_to_num(0))
#     optimizer.step()
#     del x, b, events, lmax, y, back
#     return zhat.item(), log_ess.item()


# def draw_kl_sample_estimate(
#     theta_exp: Theta,
#     estimator: Estimator,
#     df_params: DreznerFarnumBernoulliExperimentParameters,
# ) -> float:
#     x = (
#         torch.randn(
#             (df_params.tmax, df_params.kl_batch_size, df_params.fmax),
#             dtype=theta_exp[0].dtype,
#         )
#         * df_params.x_std
#     )
#     b = _psampler(x, theta_exp)
#     events, lmax = event_sample(theta_exp[2], x, b)
#     y = _pad_events(events, lmax)
#     lp = _lp(b, x, theta_exp) + _lg(y, b, x, theta_exp)
#     lq = _lp(b, x, estimator.theta) + _lg(y, b, x, estimator.theta)
#     return (lp - lq).mean().item()


# def train(
#     df_params: DreznerFarnumBernoulliExperimentParameters,
# ) -> Tuple[List[float], List[float], List[float], List[float]]:
#     optimizer, scheduler, theta_exp, theta_act, estimator = initialize(df_params)
#     sample_kls = []
#     sample_zhats = []
#     log_esss = []
#     sses_theta_1 = []
#     sses_theta_2 = []
#     sses_theta_3 = []
#     tdeltas = []
#     theta_3_exp = (
#         theta_exp[2] - theta_exp[2].mean()
#     )  # W is shift-invariant due to softmax
#     # we re-seed at the start of each trial to ensure the samples are drawn in the same
#     # way, regardless of any random draws in the algorithms themselves. N.B. There are
#     # likely fewer seeds than elements in the domain for tmax > 64, making it impossible
#     # to draw all b eventually. However, this would involve an insane number of trials
#     # anyways.
#     seeds = torch.randint(
#         torch.iinfo(torch.long).min,
#         torch.iinfo(torch.long).max,
#         (df_params.num_trials,),
#     ).tolist()
#     start = datetime.now()
#     for trial in tqdm(range(df_params.num_trials)):
#         torch.manual_seed(seeds[trial])
#         zhat, log_ess = train_for_trial(optimizer, theta_exp, estimator, df_params)
#         sample_kl = draw_kl_sample_estimate(theta_exp, estimator, df_params)
#         scheduler.step(sample_kl)
#         log_esss.append(log_ess)
#         sample_zhats.append(zhat)
#         sample_kls.append(sample_kl)
#         tdeltas.append(datetime.now() - start)
#         sses_theta_1.append(((theta_act[0] - theta_exp[0]) ** 2).item())
#         sses_theta_2.append(((theta_act[1] - theta_exp[1]) ** 2).item())
#         theta_3_act = theta_act[2] - theta_act[2].mean()
#         sses_theta_3.append(((theta_3_exp - theta_3_act) ** 2).sum().item())
#         # if not trial % 10:
#         #     print(sses_theta_1[-1], sses_theta_2[-1], sses_theta_3[-1], sample_kl)
#     return (
#         sample_kls,
#         sample_zhats,
#         log_esss,
#         sses_theta_1,
#         sses_theta_2,
#         sses_theta_3,
#         tdeltas,
#     )


# def main(args=None) -> int:
#     parser = argparse.ArgumentParser(
#         description="Run Drezner & Farnum dependent Bernoulli experiments"
#     )
#     pargparse.add_parameterized_print_group(
#         parser, DreznerFarnumBernoulliExperimentParameters
#     )
#     pargparse.add_parameterized_read_group(
#         parser, type=DreznerFarnumBernoulliExperimentParameters
#     )
#     parser.add_argument(
#         "--seed",
#         default=None,
#         type=int,
#         help="If set, overrides the seed in the config file (if any)",
#     )
#     parser.add_argument(
#         "csv",
#         nargs="?",
#         default=sys.stdout,
#         type=argparse.FileType("w"),
#         help="Where to store (csv) results to. Default is stdout",
#     )
#     options = parser.parse_args(args)
#     if options.seed is not None:
#         options.params.seed = options.seed
#     (
#         sample_kls,
#         zhats,
#         log_esss,
#         sses_theta_1,
#         sses_theta_2,
#         sses_theta_3,
#         tdeltas,
#     ) = train(options.params)
#     sses_theta_1 = pd.Series(sses_theta_1)
#     sses_theta_2 = pd.Series(sses_theta_2)
#     sses_theta_3 = pd.Series(sses_theta_3)
#     sample_kls = pd.Series(sample_kls)
#     tdeltas = pd.Series(tdeltas)
#     zhats = pd.Series(zhats)
#     trials = pd.Series(np.arange(1, options.params.num_trials + 1))
#     df = pd.DataFrame(
#         {
#             "Seed": options.params.seed,
#             "Total Number of Trials": options.params.num_trials,
#             "Training Batch Size": options.params.train_batch_size,
#             "KL Batch Size": options.params.kl_batch_size,
#             "x Sequence Length": options.params.tmax,
#             "x Vector Length": options.params.fmax,
#             "p_1": options.params.p_1,
#             "p_2": options.params.p_2,
#             "x Standard Deviation": options.params.x_std,
#             "theta_3 Standard Deviation": options.params.theta_3_std,
#             "Learning Rate": options.params.learning_rate,
#             "LR Reduction Patience": options.params.reduce_lr_patience,
#             "LR Reduction Threshold": options.params.reduce_lr_threshold,
#             "LR Reduction Factor": options.params.reduce_lr_factor,
#             "LR Reduction min": options.params.reduce_lr_min,
#             "Estimator": options.params.estimator,
#             "Optimizer": DefaultObjectSelectorSerializer().serialize(
#                 "optimizer", options.params
#             ),
#             "Number of Monte Carlo Samples": options.params.num_mc_samples,
#             "AIS Burn-in": options.params.ais_burn_in,
#             "Trial": trials,
#             "SSE theta_1": sses_theta_1,
#             "SSE theta_2": sses_theta_2,
#             "SSE theta_3": sses_theta_3,
#             "Time Since Start": tdeltas,
#             "zhat Estimate": zhats,
#             "Log ESS": log_esss,
#             "KL Batch Estimate": sample_kls,
#         }
#     )
#     df.to_csv(options.csv, header=True, index=False)
#     return 0


# def _lp(b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
#     nmax = b.size(1)
#     theta_1 = theta[0].expand(nmax)
#     theta_2 = theta[1].expand(nmax)
#     return dependent_bernoulli_logprob(theta_1, theta_2, b)


# def _lp_full(b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
#     nmax = b.size(1)
#     theta_1 = theta[0].expand(nmax)
#     theta_2 = theta[1].expand(nmax)
#     return dependent_bernoulli_logprob(theta_1, theta_2, b, True)


# def _lie_count(
#     y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
# ) -> torch.Tensor:
#     return b.log()


# @torch.jit.script
# @torch.no_grad()
# def _pad_events(events: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
#     device = events.device
#     assert device == lens.device
#     nmax = lens.size(0)
#     max_len = int(lens.max().item())
#     len_mask = lens.unsqueeze(1) > torch.arange(
#         max_len, device=device, dtype=lens.dtype
#     )
#     return torch.zeros(
#         (nmax, max_len), device=events.device, dtype=events.dtype
#     ).masked_scatter(len_mask, events)


# def _lg(
#     y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
# ) -> torch.Tensor:
#     lmax = b.sum(0)
#     len_mask = lmax.unsqueeze(1) > torch.arange(
#         y.size(1), device=lmax.device, dtype=lmax.dtype
#     )
#     events = y.masked_select(len_mask)
#     return event_logprob(theta[2], x, b, events)


# @torch.no_grad()
# def _psampler(x: torch.Tensor, theta: Theta) -> torch.Tensor:
#     tmax, nmax = x.size(0), x.size(1)
#     theta_1 = theta[0].expand(nmax)
#     theta_2 = theta[1].expand(nmax)
#     return dependent_bernoulli_sample(theta_1, theta_2, tmax)


# def _ilw(x: torch.Tensor, theta: Theta) -> torch.Tensor:
#     return theta[0].expand([x.size(0), x.size(1)])


# def _ilg(
#     x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor, theta: Theta
# ) -> torch.Tensor:
#     len_mask = lmax.unsqueeze(1) > torch.arange(
#         y.size(1), device=lmax.device, dtype=lmax.dtype
#     )
#     events = y.masked_select(len_mask)
#     return enumerate_latent_log_likelihoods(theta[2], x, events, lmax)


# if __name__ == "__main__":
#     sys.exit(main())

# # TESTS


# def test_dependent_bernoulli_expectation():
#     torch.manual_seed(1)
#     nmax, tmax = 10, 10000
#     theta_1, theta_2 = torch.randn(nmax), torch.randn(nmax)
#     b = dependent_bernoulli_sample(theta_1, theta_2, tmax)
#     p_act = b.mean(0)
#     assert torch.allclose(theta_1.sigmoid(), p_act, atol=1e-1)


# def test_dependent_bernoulli_logprob():
#     torch.manual_seed(2)
#     nmax, tmax = 100, 20
#     theta_1 = torch.randn(nmax)
#     theta_2 = torch.full((nmax,), -float("inf"))

#     # sigma(theta_2) = 0 has the same probabilities as independent Bernoulli trials
#     b = dependent_bernoulli_sample(theta_1, theta_2, tmax)
#     exp_logprob = torch.distributions.Bernoulli(logits=theta_1).log_prob(b)
#     act_logprob = dependent_bernoulli_logprob(theta_1, theta_2, b, True)
#     assert exp_logprob.shape == act_logprob.shape
#     assert torch.allclose(exp_logprob, act_logprob)

#     # sigma(theta_2) = 1 duplicates the previous Bernoulli trial's value after the first
#     theta_2 = -theta_2
#     b = dependent_bernoulli_sample(theta_1, theta_2, tmax)
#     exp_logprob = torch.distributions.Bernoulli(logits=theta_1).log_prob(b[0])
#     act_logprob = dependent_bernoulli_logprob(theta_1, theta_2, b)
#     assert exp_logprob.shape == act_logprob.shape
#     assert torch.allclose(exp_logprob, act_logprob)

#     # ensure 0-probability samples work too
#     b[1] = 1.0 - b[0]
#     act_logprob = dependent_bernoulli_logprob(theta_1, theta_2, b)
#     assert torch.all(act_logprob <= config.EPS_INF)


# def test_event_sample():
#     torch.manual_seed(3)
#     nmax, tmax, vmax = 100000, 30, 3
#     x = torch.randn((tmax, nmax, vmax))
#     b = torch.randint(0, 2, (tmax, nmax)).float()
#     theta_3 = torch.eye(vmax)
#     x[..., 0] += 1.0 - b
#     events, lens = event_sample(theta_3, x, b)
#     assert lens.sum().item() == b.sum().long().item() == events.numel()
#     freq = events.bincount() / events.numel()
#     assert freq.numel() == vmax
#     assert torch.allclose(freq, torch.tensor(1 / vmax), atol=1e-3)


# def test_event_logprob():
#     torch.manual_seed(4)
#     nmax, tmax, fmax, vmax = 10000, 30, 10, 10
#     x = torch.randn((tmax, nmax, vmax))
#     theta_3 = torch.rand(fmax, vmax)
#     b = torch.cat([torch.ones(tmax // 2, nmax), torch.zeros((tmax + 1) // 2, nmax)])
#     events, lens = event_sample(theta_3, x, b)
#     assert torch.all(lens == tmax // 2)
#     exp_logprob = (
#         (x[: tmax // 2].transpose(0, 1) @ theta_3)
#         .log_softmax(2)
#         .gather(2, events.view(nmax, tmax // 2, 1))
#         .squeeze(2)
#         .sum(1)
#     )
#     act_logprob = event_logprob(theta_3, x, b, events)
#     assert exp_logprob.size() == act_logprob.size()
#     assert torch.allclose(exp_logprob, act_logprob)


# def test_enumerate_latent_log_likelihoods():
#     torch.manual_seed(5)
#     nmax, fmax = 30, 50
#     lens = torch.arange(nmax)
#     tmax = (2 * lens.max()).item()
#     vmax = lens.sum().item()
#     events = torch.arange(vmax)
#     x = torch.randn((tmax, nmax, fmax))
#     theta_3 = torch.rand(fmax, vmax)
#     ilg = enumerate_latent_log_likelihoods(theta_3, x, events, lens)
#     assert ilg.dim() == 3
#     assert ilg.size(0) == nmax - 1
#     assert ilg.size(1) == tmax + 1
#     assert ilg.size(2) == nmax
#     for n in range(1, nmax):
#         assert lens[n] == n
#         x_n = x[:, n]
#         events_n, events = events[:n], events[n:]
#         lmax_n = n
#         assert torch.allclose(ilg[lmax_n:, :, n], torch.tensor(config.EPS_INF))
#         for ell in range(lmax_n):
#             events_n_ell = events_n[ell]
#             x_n_ell = x_n[ell : tmax - lmax_n + ell + 1]
#             probs_n_ell = (x_n_ell @ theta_3).log_softmax(1)
#             ilg_n_ell_exp = probs_n_ell[:, events_n_ell]
#             assert torch.allclose(
#                 ilg[ell, tmax - lmax_n + 1 :, n], torch.tensor(config.EPS_INF)
#             )
#             ilg_n_ell_act = ilg[ell, : tmax - lmax_n + 1, n]
#             assert ilg_n_ell_exp.size() == ilg_n_ell_act.size()
#             assert torch.allclose(ilg_n_ell_exp, ilg_n_ell_act)
#     assert not events.numel()


# def test_estimator_hooks():
#     torch.manual_seed(6)
#     tmax, nmax, fmax, vmax, mc = range(7, 12)
#     theta_1 = torch.randn((1,), requires_grad=True)
#     theta_2 = torch.randn((1,), requires_grad=True)
#     theta_3 = torch.randn((fmax, vmax), requires_grad=True)
#     theta = [theta_1, theta_2, theta_3]
#     x = torch.randn((tmax, nmax, fmax))
#     b = dependent_bernoulli_sample(theta_1.expand(nmax), theta_2.expand(nmax), tmax)
#     l1 = dependent_bernoulli_logprob(theta_1.expand(nmax), theta_2.expand(nmax), b)
#     l2 = _lp(b, x, theta)
#     assert torch.allclose(l1, l2)
#     events, lmax = event_sample(theta_3, x, b)
#     y = _pad_events(events, lmax)
#     l1 = event_logprob(theta_3, x, b, events)
#     l2 = _lg(y, b, x, theta)
#     assert torch.allclose(l1, l2)
#     l1 = enumerate_latent_log_likelihoods(theta_3, x, events, lmax)
#     l2 = _ilg(x, y, lmax, theta)
#     assert torch.allclose(l1, l2)

#     zhat, back, _ = RejectionEstimator(_psampler, mc, _lp, _lg, theta)(x, y, lmax)
#     grad_zhat = torch.autograd.grad(back, theta)
#     assert not torch.allclose(zhat, torch.tensor(0.0))
#     assert not torch.allclose(grad_zhat[0], torch.tensor(0.0))
#     assert not torch.allclose(grad_zhat[1], torch.tensor(0.0))
#     assert not torch.allclose(grad_zhat[2], torch.tensor(0.0))

#     zhat, back, _ = ExtendedConditionalBernoulliEstimator(_ilw, _ilg, _lp, _lg, theta)(
#         x, y, lmax
#     )
#     grad_zhat = torch.autograd.grad(back, theta, allow_unused=True)
#     assert not torch.allclose(zhat, torch.tensor(0.0))
#     assert not torch.allclose(grad_zhat[0], torch.tensor(0.0))
#     assert grad_zhat[1] is None  # beta, unused
#     assert not torch.allclose(grad_zhat[2], torch.tensor(0.0))


# def test_train():
#     torch.manual_seed(7)
#     nt = 100
#     df_params = DreznerFarnumBernoulliExperimentParameters(
#         num_trials=nt, tmax=16, train_batch_size=1, estimator="ecb", p_1=0.5, p_2=0
#     )
#     res = train(df_params)
#     assert (
#         torch.tensor(res[0][: nt // 2]).mean() > torch.tensor(res[0][nt // 2 :]).mean()
#     )

