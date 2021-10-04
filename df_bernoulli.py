"""Sequential classification based on Drezner & Farnum 1993"""

import torch
from torch.overrides import get_ignored_functions

torch.use_deterministic_algorithms(True)


import config
from estimators import (
    CbAisImhEstimator,
    ConditionalBernoulliEstimator,
    EnumerateEstimator,
    Estimator,
    GibbsCbAisImhEstimator,
    RejectionEstimator,
    StaticSsworImportanceSampler,
    Theta,
)
from typing import List, Tuple
import param
import argparse
import sys
import pydrobert.param.argparse as pargparse
from pydrobert.param.serialization import DefaultObjectSelectorSerializer
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime


@torch.jit.script
@torch.no_grad()
def dependent_bernoulli_sample(
    p: torch.Tensor, beta: torch.Tensor, tmax: int
) -> torch.Tensor:
    r"""Sample from Drezner & Farnum's Dependent Bernoulli model

    For a given batch element (and :math:`t` is 1-indexed)

    .. math::

        P(b_t = 1|b_{1:t-1}) = p * (1 - \beta) + \frac{\beta\sum_{t'=1}^{t-1}b_t}{t-1}

    Parameters
    ----------
    p : torch.Tensor
        Shape ``(nmax,)`` of batch's independent Bernoulli probabilities, :math:`p`
    beta : torch.Tensor
        Shape ``(nmax,)`` of batch's sample average mixing coefficient, :math:`\beta`
    tmax : int
        Number of sequential Bernoulli values to sample

    Returns
    -------
    b : torch.Tensor
        Shape ``(tmax, nmax)`` where ``b[:, n]`` is the sequence of `tmax` sampled
        Bernoulli values with parameters ``p[n]`` and ``beta[n]``.
    """
    assert tmax >= 0
    assert p.dim() == 1
    nmax = p.size(0)
    assert beta.size(0) == nmax
    device = p.device
    assert beta.device == device
    b = torch.empty((tmax, nmax), device=device, dtype=p.dtype)
    if not tmax:
        return b
    b[0] = S_tm1 = torch.bernoulli(p)
    for tm1 in range(1, tmax):
        p_t = (1.0 - beta) * p + beta * S_tm1 / tm1
        b_t = torch.bernoulli(p_t)
        b[tm1] = b_t  # 0-indexed
        S_tm1 += b_t
    return b


@torch.jit.script
def dependent_bernoulli_logprob(
    p: torch.Tensor,
    beta: torch.Tensor,
    b: torch.Tensor,
    full: bool = False,
    neg_inf: float = config.EPS_INF,
) -> torch.Tensor:
    """Return the log-probability of sample under D & F Dependent Bernoulli model

    Counterpart to :func:`dependent_bernoulli_sample`

    Returns
    -------
    logprob : torch.Tensor
        A tensor of shape ``(nmax,)`` where ``logprob[n]`` is the joint log-probability
        of the Bernoulli sequence ``b[:, n]`` under the Drezner & Farnum model.
    """
    assert b.dim() == 2
    assert beta.size() == p.size() == b.size()[1:]
    tmax, device = b.size(0), b.device
    assert p.device == beta.device == device
    if not tmax:
        return torch.empty((0, beta.size(0)), device=device, dtype=p.dtype)
    b = b.detach()
    logp, log1mp = p.log(), torch.log1p(-p)
    logp_1 = (logp * b[0]).nan_to_num_(neginf=neg_inf) + (
        log1mp * (1.0 - b[0])
    ).nan_to_num_(neginf=neg_inf)
    if tmax == 1:
        return logp_1
    S_tm1 = b[:-1].cumsum(0)
    t_tm1 = torch.arange(1, tmax, device=device).unsqueeze(1)
    logt_tm1 = t_tm1.log()
    logpp_t, log1mpp_t = S_tm1.log() - logt_tm1, (t_tm1 - S_tm1).log() - logt_tm1
    logbeta, log1mbeta = beta.log(), torch.log1p(-beta)
    b, negb = b[1:], 1.0 - b[1:]
    left_t = (
        (logp.unsqueeze(0) * b).nan_to_num_(neginf=neg_inf)
        + (log1mp.unsqueeze(0) * negb).nan_to_num_(neginf=neg_inf)
        + log1mbeta
    )
    right_t = (
        (logpp_t * b).nan_to_num_(neginf=neg_inf)
        + (log1mpp_t * negb).nan_to_num_(neginf=neg_inf)
        + logbeta
    )
    max_t = torch.max(left_t, right_t).nan_to_num_()
    logp_t = ((left_t - max_t).exp() + (right_t - max_t).exp()).log() + max_t
    if full:
        return torch.cat([logp_1.unsqueeze(0), logp_t])
    else:
        return logp_1 + logp_t.sum(0)


@torch.jit.script
@torch.no_grad()
def event_sample(
    W: torch.Tensor, x: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert b.dim() == W.dim() == 2
    assert x.dim() == 3
    assert x.size()[:2] == b.size()
    isize = x.size(2)
    assert isize == W.size(0)
    assert b.device == W.device == x.device
    lens = b.to(torch.long).sum(0)
    x_at_b = (
        x.transpose(0, 1)
        .masked_select(b.to(torch.bool).t().unsqueeze(2))
        .view(-1, isize)
    )
    logits = x_at_b @ W
    probs = logits.softmax(1)
    y = torch.multinomial(probs, 1, True).flatten()
    return y, lens


# @torch.jit.script
def event_logprob(
    W: torch.Tensor, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Return the log-probability of sample y given b and x

    Counterpart to :func:`event_sample`.

    Returns
    -------
    logprob : torch.Tensor
        Of shape ``(nmax,)`` where ``logprob[n]`` is the joint log-probability of the
        sequence ``y[lens[n - 1]:lens[n]]`` given ``x[..., n]``, ``b[:, n]``, and
        parameters ``W[n]``.
    """
    assert b.dim() == W.dim() == 2
    assert x.dim() == 3
    assert y.dim() == 1
    assert x.size()[:2] == b.size()
    isize = x.size(2)
    assert isize == W.size(0)
    assert b.device == W.device == x.device == y.device
    b_ = b.t() != 0
    x_at_b = x.transpose(0, 1).masked_select(b_.unsqueeze(2)).view(-1, isize)
    assert y.numel() == x_at_b.size(0), (
        y.numel(),
        x_at_b.size(),
        b.size(),
        b.sum(),
        b[:, 0],
        b_[0],
    )
    logits = x_at_b @ W
    logprob = logits.log_softmax(1).gather(1, y.unsqueeze(1)).squeeze(1)
    return b.t().masked_scatter(b_, logprob).sum(1)


@torch.jit.script
def enumerate_latent_log_likelihoods(
    W: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    lens: torch.Tensor,
    neg_inf: float = config.EPS_INF,
) -> torch.Tensor:
    assert W.dim() == 2
    assert x.dim() == 3
    assert y.dim() == lens.dim() == 1
    device = W.device
    assert device == x.device == y.device == lens.device
    numel = torch.tensor(y.numel(), device=device)
    assert numel == lens.sum()
    limits = lens.cumsum(0)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), limits[:-1]])
    tmax, nmax, fmax = x.size()
    lens_min, lens_max = int(lens.min().long().item()), int(lens.max().item())
    x = torch.cat(
        [x, torch.zeros((tmax - lens_min + lens_max + 1, nmax, fmax), device=device)]
    )
    if lens_max == 0:
        return torch.zeros((0, tmax - lens_min + 1, nmax), device=device, dtype=x.dtype)
    probs = (x @ W).log_softmax(2)
    g = []
    for ell in range(lens_max):
        y_ell = y[(offsets + ell).clamp_max_(numel - 1)]
        probs_ell = probs[ell : tmax - lens_min + ell + 1]
        g.append(
            probs_ell.gather(
                2, y_ell.expand(probs_ell.size()[:-1]).unsqueeze(2)
            ).squeeze(2)
        )
    mask = (
        torch.arange(tmax - lens_min + 1, device=device).unsqueeze(1) >= tmax - lens + 1
    )
    mask = (torch.arange(lens_max).unsqueeze(1) >= lens).unsqueeze(1) | mask
    return torch.stack(g, 0).masked_fill(mask, neg_inf)


# For command-line task


class DreznerFarnumBernoulliExperimentParameters(param.Parameterized):
    seed = param.Integer(
        None,
        bounds=(-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF),
        inclusive_bounds=(True, True),
    )
    num_trials = param.Integer(1024, bounds=(1, None))
    train_batch_size = param.Integer(1, bounds=(1, None))
    kl_batch_size = param.Integer(128, bounds=(1, None))
    tmax = param.Integer(128, bounds=(1, None))
    fmax = param.Integer(16, bounds=(1, None))
    vmax = param.Integer(16, bounds=(1, None))
    p = param.Magnitude(None)
    beta = param.Magnitude(None)
    x_std = param.Number(1.0, bounds=(0, None))
    W_std = param.Number(1.0, bounds=(0, None))
    learning_rate = param.Magnitude(1e-3)
    estimator = param.ObjectSelector(
        "rej", objects=("rej", "enum", "sswor", "cb", "ais-cb-lp", "ais-cb-gibbs"),
    )
    optimizer = param.ObjectSelector(
        torch.optim.Adam, objects={"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    )
    num_mc_samples = param.Integer(2 ** 8, bounds=(1, None))


def initialize(
    df_params: DreznerFarnumBernoulliExperimentParameters,
) -> Tuple[torch.optim.Optimizer, Theta, Theta, Estimator]:
    dtype = torch.float
    if df_params.seed is not None:
        torch.manual_seed(df_params.seed)
    if df_params.p is None:
        df_params.p = torch.randn(1, dtype=torch.double).sigmoid_().item()
    lp_exp = torch.tensor(df_params.p, dtype=dtype).logit_()
    if df_params.beta is None:
        df_params.beta = torch.randn(1, dtype=torch.double).sigmoid_().item()
    lbeta_exp = torch.tensor(df_params.beta, dtype=dtype).logit_()
    W_exp = torch.randn((df_params.fmax, df_params.vmax), dtype=dtype) * df_params.W_std
    theta_exp = [lp_exp, lbeta_exp, W_exp]
    lp_act = torch.randn(1, dtype=dtype).requires_grad_(True)
    lbeta_act = torch.randn(1, dtype=dtype).requires_grad_(True)
    W_act = torch.randn(
        (df_params.fmax, df_params.vmax), dtype=dtype, requires_grad=True
    )
    theta_act = [lp_act, lbeta_act, W_act]
    optim_params = theta_act.copy()
    if df_params.beta == 0.0 or df_params.estimator == "cb":
        lbeta_act.data[0] = -float("inf")
        optim_params.pop(1)
    if df_params.estimator == "rej":
        estimator = RejectionEstimator(
            _psampler, df_params.num_mc_samples, _lp, _lg, theta_act
        )
    elif df_params.estimator == "enum":
        estimator = EnumerateEstimator(_lp, _lg, theta_act)
    elif df_params.estimator == "sswor":
        estimator = StaticSsworImportanceSampler(
            df_params.num_mc_samples, _lp, _lg, theta_act
        )
    elif df_params.estimator == "cb":
        estimator = ConditionalBernoulliEstimator(_ilw, _ilg, _lp, _lg, theta_act)
    elif df_params.estimator == "ais-cb-lp":
        estimator = CbAisImhEstimator(
            _lie_lp_only,
            df_params.num_mc_samples,
            _lp,
            _lg,
            theta_act,
            li_requires_norm=False,
        )
    elif df_params.estimator == "ais-cb-gibbs":
        estimator = GibbsCbAisImhEstimator(
            df_params.num_mc_samples, _lp, _lg, theta_act
        )
    else:
        raise NotImplementedError
    optimizer = df_params.optimizer(optim_params, lr=df_params.learning_rate)
    return optimizer, theta_exp, theta_act, estimator


def train_for_trial(
    optimizer: torch.optim.Optimizer,
    theta_exp: Theta,
    estimator: Estimator,
    df_params: DreznerFarnumBernoulliExperimentParameters,
) -> float:
    optimizer.zero_grad()
    x = (
        torch.randn(
            (df_params.tmax, df_params.train_batch_size, df_params.fmax),
            dtype=theta_exp[0].dtype,
        )
        * df_params.x_std
    )
    b = _psampler(x, theta_exp)
    events, lmax = event_sample(theta_exp[2], x, b)
    y = _pad_events(events, lmax)
    zhat, back = estimator(x, y, lmax)
    (-back).backward()
    # grads_theta = torch.autograd.grad(-back, estimator.theta, allow_unused=True)
    # for v, g in zip(estimator.theta, grads_theta):
    #     if g is not None:
    #         v.backward(g.nan_to_num(0))
    optimizer.step()
    del x, b, events, lmax, y, back
    return zhat.item()


def draw_kl_sample_estimate(
    theta_exp: Theta,
    estimator: Estimator,
    df_params: DreznerFarnumBernoulliExperimentParameters,
) -> float:
    x = (
        torch.randn(
            (df_params.tmax, df_params.kl_batch_size, df_params.fmax),
            dtype=theta_exp[0].dtype,
        )
        * df_params.x_std
    )
    b = _psampler(x, theta_exp)
    events, lmax = event_sample(theta_exp[2], x, b)
    y = _pad_events(events, lmax)
    lp = _lp(b, x, theta_exp) + _lg(y, b, x, theta_exp)
    lq = _lp(b, x, estimator.theta) + _lg(y, b, x, estimator.theta)
    return (lp - lq).mean().item()


def train(
    df_params: DreznerFarnumBernoulliExperimentParameters,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    optimizer, theta_exp, theta_act, estimator = initialize(df_params)
    sample_kls = []
    sample_zhats = []
    sses_lp = []
    sses_lbeta = []
    sses_W = []
    tdeltas = []
    lp_exp = theta_exp[0].sigmoid().log().item()
    lbeta_exp = theta_exp[1].sigmoid().log().item()
    start = datetime.now()
    for _ in tqdm(range(df_params.num_trials)):
        sample_zhats.append(train_for_trial(optimizer, theta_exp, estimator, df_params))
        sample_kls.append(draw_kl_sample_estimate(theta_exp, estimator, df_params))
        tdeltas.append(datetime.now() - start)
        sses_lp.append((theta_act[0].sigmoid().log().item() - lp_exp) ** 2)
        sses_lbeta.append((theta_act[1].sigmoid().log().item() - lbeta_exp) ** 2)
        sses_W.append(((theta_exp[2] - theta_act[2]) ** 2).sum().item())
    return sample_kls, sample_zhats, sses_lp, sses_lbeta, sses_W, tdeltas


def main(args=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Drezner & Farnum dependent Bernoulli experiments"
    )
    pargparse.add_parameterized_print_group(
        parser, DreznerFarnumBernoulliExperimentParameters
    )
    pargparse.add_parameterized_read_group(
        parser, type=DreznerFarnumBernoulliExperimentParameters
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="If set, overrides the seed in the config file (if any)",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=sys.stdout,
        type=argparse.FileType("w"),
        help="Where to store (csv) results to. Default is stdout",
    )
    options = parser.parse_args(args)
    if options.seed is not None:
        options.params.seed = options.seed
    sample_kls, zhats, sses_lp, sses_lbeta, sses_W, tdeltas = train(options.params)
    sses_lp = pd.Series(sses_lp)
    sses_lbeta = pd.Series(sses_lbeta)
    sses_W = pd.Series(sses_W)
    sample_kls = pd.Series(sample_kls)
    tdeltas = pd.Series(tdeltas)
    zhats = pd.Series(zhats)
    trials = pd.Series(np.arange(1, options.params.num_trials + 1))
    df = pd.DataFrame(
        {
            "Seed": np.nan if options.params.seed is None else options.params.seed,
            "Total Number of Trials": options.params.num_trials,
            "Training Batch Size": options.params.train_batch_size,
            "KL Batch Size": options.params.kl_batch_size,
            "x Sequence Length": options.params.tmax,
            "x Vector Length": options.params.fmax,
            "p": np.nan if options.params.p is None else options.params.p,
            "beta": np.nan if options.params.beta is None else options.params.beta,
            "x Standard Deviation": options.params.x_std,
            "W Standard Deviation": options.params.W_std,
            "Learning Rate": options.params.learning_rate,
            "Estimator": options.params.estimator,
            "Optimizer": DefaultObjectSelectorSerializer().serialize(
                "optimizer", options.params
            ),
            "Number of Monte Carlo Samples": options.params.num_mc_samples,
            "Trial": trials,
            "SSE Log p": sses_lp,
            "SSE Log beta": sses_lbeta,
            "SSE W": sses_W,
            "Time Since Start": tdeltas,
            "zhat Estimate": zhats,
            "KL Batch Estimate": sample_kls,
        }
    )
    df.to_csv(options.csv, header=True, index=False)
    return 0


def _lp(b: torch.Tensor, x: torch.Tensor, theta: Theta) -> torch.Tensor:
    p = theta[0].sigmoid()
    beta = theta[1].sigmoid()
    nmax = b.size(1)
    p = p.expand(nmax)
    beta = beta.expand(nmax)
    return dependent_bernoulli_logprob(p, beta, b)


def _lie_lp_only(
    y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
) -> torch.Tensor:
    # this version of the log-inclusion probability estimator is biased, simply using
    # the conditional log probabilities when computing log P(b)
    p = theta[0].sigmoid()
    beta = theta[1].sigmoid()
    nmax = b.size(1)
    p = p.expand(nmax)
    beta = beta.expand(nmax)
    return dependent_bernoulli_logprob(p, beta, b, True)


@torch.jit.script
@torch.no_grad()
def _pad_events(events: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    device = events.device
    assert device == lens.device
    nmax = lens.size(0)
    max_len = int(lens.max().item())
    len_mask = lens.unsqueeze(1) > torch.arange(
        max_len, device=device, dtype=lens.dtype
    )
    return torch.zeros(
        (nmax, max_len), device=events.device, dtype=events.dtype
    ).masked_scatter(len_mask, events)


def _lg(
    y: torch.Tensor, b: torch.Tensor, x: torch.Tensor, theta: Theta
) -> torch.Tensor:
    W = theta[2]
    lmax = b.sum(0)
    len_mask = lmax.unsqueeze(1) > torch.arange(
        y.size(1), device=lmax.device, dtype=lmax.dtype
    )
    events = y.masked_select(len_mask)
    return event_logprob(W, x, b, events)


@torch.no_grad()
def _psampler(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    p = theta[0].sigmoid()
    beta = theta[1].sigmoid()
    tmax, nmax = x.size(0), x.size(1)
    p = p.expand(nmax)
    beta = beta.expand(nmax)
    return dependent_bernoulli_sample(p, beta, tmax)


def _ilw(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    logits = theta[0]
    return logits.expand([x.size(0), x.size(1)])


def _ilg(
    x: torch.Tensor, y: torch.Tensor, lmax: torch.Tensor, theta: Theta
) -> torch.Tensor:
    W = theta[2]
    len_mask = lmax.unsqueeze(1) > torch.arange(
        y.size(1), device=lmax.device, dtype=lmax.dtype
    )
    events = y.masked_select(len_mask)
    return enumerate_latent_log_likelihoods(W, x, events, lmax)


if __name__ == "__main__":
    sys.exit(main())

# TESTS


def test_dependent_bernoulli_expectation():
    torch.manual_seed(1)
    nmax, tmax = 10, 10000
    # beta close to 1 take a long time to converge
    p, beta = torch.rand(nmax), torch.rand(nmax) * 0.5 + 0.25
    b = dependent_bernoulli_sample(p, beta, tmax)
    p_act = b.mean(0)
    assert torch.allclose(p, p_act, atol=1e-1)


def test_dependent_bernoulli_logprob():
    torch.manual_seed(2)
    nmax, tmax = 100, 20
    p = torch.rand(nmax)

    # beta = 0 has the same probabilities as independent Bernoulli trials
    b = dependent_bernoulli_sample(p, torch.zeros((nmax,)), tmax)
    exp_logprob = torch.distributions.Bernoulli(probs=p).log_prob(b)
    act_logprob = dependent_bernoulli_logprob(p, torch.zeros((nmax,)), b, True)
    assert exp_logprob.shape == act_logprob.shape
    assert torch.allclose(exp_logprob, act_logprob)

    # beta = 1 duplicates the previous Bernoulli trial's value after the first trial
    b = dependent_bernoulli_sample(p, torch.ones((nmax,)), tmax)
    exp_logprob = torch.distributions.Bernoulli(probs=p).log_prob(b[0])
    act_logprob = dependent_bernoulli_logprob(p, torch.ones((nmax,)), b)
    assert exp_logprob.shape == act_logprob.shape
    assert torch.allclose(exp_logprob, act_logprob)

    # ensure 0-probability samples work too
    b[1] = 1.0 - b[0]
    act_logprob = dependent_bernoulli_logprob(p, torch.ones((nmax,)), b)
    assert torch.all(act_logprob <= config.EPS_INF)


def test_event_sample():
    torch.manual_seed(3)
    nmax, tmax, vmax = 100000, 30, 3
    x = torch.randn((tmax, nmax, vmax))
    b = torch.randint(0, 2, (tmax, nmax)).float()
    W = torch.eye(vmax)
    x[..., 0] += 1.0 - b
    events, lens = event_sample(W, x, b)
    assert lens.sum().item() == b.sum().long().item() == events.numel()
    freq = events.bincount() / events.numel()
    assert freq.numel() == vmax
    assert torch.allclose(freq, torch.tensor(1 / vmax), atol=1e-3)


def test_event_logprob():
    torch.manual_seed(4)
    nmax, tmax, fmax, vmax = 10000, 30, 10, 10
    x = torch.randn((tmax, nmax, vmax))
    W = torch.rand(fmax, vmax)
    b = torch.cat([torch.ones(tmax // 2, nmax), torch.zeros((tmax + 1) // 2, nmax)])
    events, lens = event_sample(W, x, b)
    assert torch.all(lens == tmax // 2)
    exp_logprob = (
        (x[: tmax // 2].transpose(0, 1) @ W)
        .log_softmax(2)
        .gather(2, events.view(nmax, tmax // 2, 1))
        .squeeze(2)
        .sum(1)
    )
    act_logprob = event_logprob(W, x, b, events)
    assert exp_logprob.size() == act_logprob.size()
    assert torch.allclose(exp_logprob, act_logprob)


def test_enumerate_latent_log_likelihoods():
    torch.manual_seed(5)
    nmax, fmax = 30, 50
    lens = torch.arange(nmax)
    tmax = (2 * lens.max()).item()
    vmax = lens.sum().item()
    events = torch.arange(vmax)
    x = torch.randn((tmax, nmax, fmax))
    W = torch.rand(fmax, vmax)
    ilg = enumerate_latent_log_likelihoods(W, x, events, lens)
    assert ilg.dim() == 3
    assert ilg.size(0) == nmax - 1
    assert ilg.size(1) == tmax + 1
    assert ilg.size(2) == nmax
    for n in range(1, nmax):
        assert lens[n] == n
        x_n = x[:, n]
        events_n, events = events[:n], events[n:]
        lmax_n = n
        assert torch.allclose(ilg[lmax_n:, :, n], torch.tensor(config.EPS_INF))
        for ell in range(lmax_n):
            events_n_ell = events_n[ell]
            x_n_ell = x_n[ell : tmax - lmax_n + ell + 1]
            probs_n_ell = (x_n_ell @ W).log_softmax(1)
            ilg_n_ell_exp = probs_n_ell[:, events_n_ell]
            assert torch.allclose(
                ilg[ell, tmax - lmax_n + 1 :, n], torch.tensor(config.EPS_INF)
            )
            ilg_n_ell_act = ilg[ell, : tmax - lmax_n + 1, n]
            assert ilg_n_ell_exp.size() == ilg_n_ell_act.size()
            assert torch.allclose(ilg_n_ell_exp, ilg_n_ell_act)
    assert not events.numel()


def test_estimator_hooks():
    torch.manual_seed(6)
    tmax, nmax, fmax, vmax, mc = range(7, 12)
    p = torch.randn((1,)).sigmoid().requires_grad_(True)
    beta = torch.randn((1,)).sigmoid().requires_grad_(True)
    W = torch.randn((fmax, vmax), requires_grad=True)
    theta = [p.logit(), beta.logit(), W]
    x = torch.randn((tmax, nmax, fmax))
    b = dependent_bernoulli_sample(p.expand(nmax), beta.expand(nmax), tmax)
    l1 = dependent_bernoulli_logprob(p.expand(nmax), beta.expand(nmax), b)
    l2 = _lp(b, x, theta)
    assert torch.allclose(l1, l2)
    events, lmax = event_sample(W, x, b)
    y = _pad_events(events, lmax)
    l1 = event_logprob(W, x, b, events)
    l2 = _lg(y, b, x, theta)
    assert torch.allclose(l1, l2)
    l1 = enumerate_latent_log_likelihoods(W, x, events, lmax)
    l2 = _ilg(x, y, lmax, theta)
    assert torch.allclose(l1, l2)

    zhat, back = RejectionEstimator(_psampler, mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat = torch.autograd.grad(back, theta)
    assert not torch.allclose(zhat, torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[0], torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[1], torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[2], torch.tensor(0.0))

    zhat, back = ConditionalBernoulliEstimator(_ilw, _ilg, _lp, _lg, theta)(x, y, lmax)
    grad_zhat = torch.autograd.grad(back, theta, allow_unused=True)
    assert not torch.allclose(zhat, torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[0], torch.tensor(0.0))
    assert grad_zhat[1] is None  # beta, unused
    assert not torch.allclose(grad_zhat[2], torch.tensor(0.0))

    zhat, back = CbAisImhEstimator(_lie_lp_only, mc, _lp, _lg, theta)(x, y, lmax)
    grad_zhat = torch.autograd.grad(back, theta, allow_unused=True)
    assert not torch.allclose(zhat, torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[0], torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[1], torch.tensor(0.0))
    assert not torch.allclose(grad_zhat[2], torch.tensor(0.0))


def test_train():
    torch.manual_seed(7)
    df_params = DreznerFarnumBernoulliExperimentParameters(
        num_trials=5, tmax=16, train_batch_size=32, estimator="enum"
    )
    res = train(df_params)
    assert res[0][0] > res[0][-1]
