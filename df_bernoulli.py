"""Sequential classification based on Drezner & Farnum 1993"""

from typing import Tuple
import torch


@torch.jit.script
@torch.no_grad()
def dependent_bernoulli_sample(
    p: torch.Tensor, theta: torch.Tensor, tmax: int
) -> torch.Tensor:
    r"""Sample from Drezner & Farnum's Dependent Bernoulli model

    For a given batch element (and :math:`t` is 1-indexed)

    .. math::

        P(b_t = 1|b_{1:t-1}) = p * (1 - \theta) + \frac{\theta\sum_{t'=1}^{t-1}b_t}{t-1}

    Parameters
    ----------
    p : torch.Tensor
        Shape ``(nmax,)`` of batch's independent Bernoulli probabilities, :math:`p`
    theta : torch.Tensor
        Shape ``(nmax,)`` of batch's sample average mixing coefficient, :math:`\theta`
    tmax : int
        Number of sequential Bernoulli values to sample

    Returns
    -------
    b : torch.Tensor
        Shape ``(tmax, nmax)`` where ``b[:, n]`` is the sequence of `tmax` sampled
        Bernoulli values with parameters ``p[n]`` and ``theta[n]``.
    """
    assert tmax >= 0
    assert p.dim() == 1
    nmax = p.size(0)
    assert theta.size(0) == nmax
    device = p.device
    assert theta.device == device
    b = torch.empty((tmax, nmax), device=device)
    if not tmax:
        return b
    b[0] = S_tm1 = torch.bernoulli(p)
    for tm1 in range(1, tmax):
        p_t = (1.0 - theta) * p + theta * S_tm1 / tm1
        b_t = torch.bernoulli(p_t)
        b[tm1] = b_t  # 0-indexed
        S_tm1 += b_t
    return b


@torch.jit.script
def dependent_bernoulli_logprob(
    p: torch.Tensor, theta: torch.Tensor, b: torch.Tensor
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
    assert theta.size() == p.size() == b.size()[1:]
    tmax, device = b.size(0), b.device
    assert p.device == theta.device == device
    if not tmax:
        return torch.empty((0, theta.size(0)), device=device)
    b = b.detach()
    logp, log1mp = p.log(), torch.log1p(-p)
    logp_1 = (logp * b[0]).nan_to_num_(neginf=-float("inf")) + (
        log1mp * (1.0 - b[0])
    ).nan_to_num_(neginf=-float("inf"))
    if tmax == 1:
        return logp_1
    S_tm1 = b[:-1].cumsum(0)
    t_tm1 = torch.arange(1, tmax, device=device).unsqueeze(1)
    logt_tm1 = t_tm1.log()
    logpp_t, log1mpp_t = S_tm1.log() - logt_tm1, (t_tm1 - S_tm1).log() - logt_tm1
    logtheta, log1mtheta = theta.log(), torch.log1p(-theta)
    b, negb = b[1:], 1.0 - b[1:]
    left_t = (
        (logp.unsqueeze(0) * b).nan_to_num_(neginf=-float("inf"))
        + (log1mp.unsqueeze(0) * negb).nan_to_num_(neginf=-float("inf"))
        + log1mtheta
    )
    right_t = (
        (logpp_t * b).nan_to_num_(neginf=-float("inf"))
        + (log1mpp_t * negb).nan_to_num_(neginf=-float("inf"))
        + logtheta
    )
    max_t = torch.max(left_t, right_t).nan_to_num_()
    logp_t = ((left_t - max_t).exp() + (right_t - max_t).exp()).log() + max_t
    return logp_1 + logp_t.sum(0)


@torch.jit.script
@torch.no_grad()
def event_sample(
    W: torch.Tensor, x: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Sample events with categorical model of y conditioned on b and x

    For a given batch element (:math:`t` and :math:`\ell` are 1-indexed)

    .. math::
        P(\tau_\ell = v|b,x) = \frac{exp(Wx_{\tau_\ell})_v}
                                    {\sum_{v'} exp(Wx_{\tau_\ell})_{v'}}
        \tau_\ell = \min_t \{t : \sum_{t'=1}^t b_t = \ell\}

    Parameters
    ----------
    W : torch.Tensor
        A matrix of shape ``(fmax, vmax)``, :math:`W`
    x : torch.Tensor
        Of shape ``(tmax, nmax, fmax)``, :math:`x`
    b : torch.Tensor
        Of shape ``(tmax, nmax)``, :math:`b`

    Returns
    -------
    y, lens : torch.Tensor, torch.Tensor
        `y` is a long tensor of shape ``(total_num_events,)`` where each element is an
        index between ``[0, vmax)`` representing the class label :math:`v` for
        some :math:`\tau_\ell`. `lens` is a long tensor of shape ``(nmax,)`` of the
        number of events per batch element of `b`. The values ``y[:lens[0]]`` correspond
        to the label sequence for Bernoulli sequence ``b[:, 0]``, ``y[lens[0]:lens[1]]``
        for ``b[:, 1]``, and so on.
    """
    assert b.dim() == W.dim() == 2
    assert x.dim() == 3
    assert x.size()[:2] == b.size()
    isize = x.size(2)
    assert isize == W.size(0)
    assert b.device == W.device == x.device
    lens = b.to(torch.long).sum(0)
    x_at_b = x.masked_select(b.to(torch.bool).unsqueeze(2)).view(-1, isize)
    logits = x_at_b @ W
    probs = logits.softmax(1)
    y = torch.multinomial(probs, 1).flatten()
    return y, lens


@torch.jit.script
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
    x_at_b = x.masked_select(b.to(torch.bool).unsqueeze(2)).view(-1, isize)
    assert y.numel() == x_at_b.size(0)
    logits = x_at_b @ W
    logprob = logits.log_softmax(1).gather(1, y.unsqueeze(1)).squeeze(1)
    return b.masked_scatter(b.to(torch.bool), logprob).sum(0)


@torch.jit.script
def enumerate_latent_likelihoods(
    W: torch.Tensor, x: torch.Tensor, y: torch.Tensor, lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Enumerate the likelihood over b of y given b and x

    For use on `y` and `lens` returned from :func:`event_sample`. Rather than returning
    the log probability of entire sequences :math:`y` conditioned on a fixed latent
    :math:`b` (as in :func:`event_logprob`), this function enumerates the probability
    of :math:`y_\ell` depending on the corresponding :math:`\tau_\ell`. The return
    value ``g[ell, t', n]``, which we denote as :math:`g_{t'}^{(\ell)}` for fixed batch
    index ``n``, has the interpretation

    .. math::

        g_{t'}^{(\ell)} = P(y_\ell|\tau_\ell = \ell + t', x)

    Note that the index ``t'`` is shifted left of the index :math:`t` as
    :math:`\tau_\ell < t` is impossible.

    Returns
    -------
    g, lens_ : torch.Tensor `g` is a tensor of shape ``(lens_max, tmax - lens_min + 1,
        nmax)`` containing the values of :math:`g_{t'}^{(\ell)}` and `lens_` is a tensor
        of shape ``(nmax,)`` containing the underlying lengths of the subsequences of
        `g` in the second dimension. ``lens_max == lens.max()`` and ``lens_min ==
        lens.min()``. An individual batch element ``n`` has no more than ``(lens[n],
        tmax - lens[n] + 1)`` values of :math:`g_{t'}^{(\ell)}`, but `g` has been padded
        along the first and second dimension so that all batch elements' values of
        :math:`g_{t'}^{(\ell)}` can fit in `g`. Thus only the values of ``g[:lens[n],
        tmax - lens[n] + 1, n]`` are well-defined for fixed ``n``.
    """
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
        [x, torch.empty((tmax - lens_min + lens_max + 1, nmax, fmax), device=device)]
    )
    probs = (x @ W).softmax(2)
    g = []
    for ell in range(lens_max):
        y_ell = y[(offsets + ell).clamp_max_(numel - 1)]
        probs_ell = probs[ell : tmax - lens_min + ell + 1]
        g.append(
            probs_ell.gather(
                2, y_ell.view(1, nmax, 1).expand(tmax - lens_min + 1, nmax, 1)
            ).squeeze(2)
        )
    return torch.stack(g, 0), tmax - lens


# TESTS


def test_dependent_bernoulli_expectation():
    torch.manual_seed(1)
    nmax, tmax = 10, 10000
    # theta close to 1 take a long time to converge
    p, theta = torch.rand(nmax), torch.rand(nmax) * 0.5 + 0.25
    b = dependent_bernoulli_sample(p, theta, tmax)
    p_act = b.mean(0)
    assert torch.allclose(p, p_act, atol=1e-1)


def test_dependent_bernoulli_logprob():
    torch.manual_seed(2)
    nmax, tmax = 100, 20
    p = torch.rand(nmax)

    # theta = 0 has the same probabilities as independent Bernoulli trials
    b = dependent_bernoulli_sample(p, torch.zeros((nmax,)), tmax)
    exp_logprob = torch.distributions.Bernoulli(probs=p).log_prob(b).sum(0)
    act_logprob = dependent_bernoulli_logprob(p, torch.zeros((nmax,)), b)
    assert exp_logprob.shape == act_logprob.shape
    assert torch.allclose(exp_logprob, act_logprob)

    # theta = 1 duplicates the previous Bernoulli trial's value after the first trial
    b = dependent_bernoulli_sample(p, torch.ones((nmax,)), tmax)
    exp_logprob = torch.distributions.Bernoulli(probs=p).log_prob(b[0])
    act_logprob = dependent_bernoulli_logprob(p, torch.ones((nmax,)), b)
    assert exp_logprob.shape == act_logprob.shape
    assert torch.allclose(exp_logprob, act_logprob)

    # ensure 0-probability samples work too
    b[1] = 1.0 - b[0]
    act_logprob = dependent_bernoulli_logprob(p, torch.ones((nmax,)), b)
    assert torch.all(act_logprob.isinf())


def test_event_sample():
    torch.manual_seed(3)
    nmax, tmax, vmax = 10000, 30, 3
    x = torch.randn((tmax, nmax, vmax))
    b = torch.randint(0, 2, (tmax, nmax)).float()
    W = torch.eye(vmax)
    x[..., 0] += 1.0 - b
    y, lens = event_sample(W, x, b)
    assert lens.sum().item() == b.sum().long().item() == y.numel()
    freq = y.bincount() / y.numel()
    assert freq.numel() == vmax
    assert torch.allclose(freq, torch.tensor(1 / vmax), atol=1e-3)


def test_event_logprob():
    torch.manual_seed(4)
    nmax, tmax, fmax, vmax = 10000, 30, 10, 10
    x = torch.randn((tmax, nmax, vmax))
    W = torch.rand(fmax, vmax)
    b = torch.cat([torch.ones(tmax // 2, nmax), torch.zeros((tmax + 1) // 2, nmax)])
    y, lens = event_sample(W, x, b)
    assert torch.all(lens == tmax // 2)
    exp_logprob = (
        (x[: tmax // 2] @ W)
        .log_softmax(2)
        .gather(2, y.view(tmax // 2, nmax, 1))
        .squeeze(2)
        .sum(0)
    )
    act_logprob = event_logprob(W, x, b, y)
    assert exp_logprob.size() == act_logprob.size()
    assert torch.allclose(exp_logprob, act_logprob)


def test_enumerate_latent_likelihoods():
    torch.manual_seed(5)
    nmax, fmax = 30, 50
    lens = torch.arange(nmax)
    tmax = (2 * lens.max()).item()
    vmax = lens.sum().item()
    y = torch.arange(vmax)
    x = torch.randn((tmax, nmax, fmax))
    W = torch.rand(fmax, vmax)
    g, lens_ = enumerate_latent_likelihoods(W, x, y, lens)
    assert torch.all(lens_ == tmax - lens)
    assert g.dim() == 3
    assert g.size(0) == nmax - 1
    assert g.size(1) == tmax + 1
    assert g.size(2) == nmax
    for n in range(1, nmax):
        assert lens[n] == n
        x_n = x[:, n]
        y_n, y = y[:n], y[n:]
        lmax_n = n
        for ell in range(lmax_n):
            y_n_ell = y_n[ell]
            x_n_ell = x_n[ell : tmax - lmax_n + ell + 1]
            probs_n_ell = (x_n_ell @ W).softmax(1)
            g_n_ell_exp = probs_n_ell[:, y_n_ell]
            g_n_ell_act = g[ell, : tmax - lmax_n + 1, n]
            assert g_n_ell_exp.size() == g_n_ell_act.size()
            assert torch.allclose(g_n_ell_exp, g_n_ell_act)
    assert not y.numel()
