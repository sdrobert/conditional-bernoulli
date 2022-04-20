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

"""Forward and backward functions"""

import torch

import pydrobert.torch.config as config


@torch.jit.script_if_tracing
def extract_relevant_odds_forward(
    w: torch.Tensor, lmax: torch.Tensor, batch_first: bool = False, fill: float = 0.0
) -> torch.Tensor:
    r"""Transforms a tensor of odds into a shape usable for forward comps

    In the forward algorithm, only a subset of odds with index :math:`t` will be
    relevant when determining the :math:`\ell`-th event location when the total number
    of events are fixed:

    * :math:`t \geq \ell`, since it is impossible to have seen :math`\ell - 1` events
      prior to the :math:`\ell`-th possible event location
    * :math:`t \leq tmax - lmax + \ell`, since it is impossible to see
      :math:`\lmax - \ell + 1` events in the future with only :math:`\lmax - \ell`
      upcoming event locations.

    This function rearranges `w` to index only the relevant odds for a given event
    location :math:`\ell`.

    Parameters
    ----------
    w : torch.Tensor
        Of shape ``(binom(tmax, kmax), nmax)``. ``w[:, n]`` are the weights for batch
        element ``n``. For ``kmax == 1``, the first index specifies the ``t``-th
        independent odds. Otherwise, the first index is a flattened, column-major
        multi-index of length ``kmax``, ``bart[1:kmax]``, such that
        ``0 <= bart[0] < bart[1] < ... < bart[kmax] < tmax`` and ``w[bart, n]`` are
        the odds of the next event location being ``bart[kmax]`` given
        prior event locations ``bart[:kmax - 1]``.
    lmax : torch.Tensor
        Of shape ``(nmax,)`` where ``lmax[n]`` is the target number of events for
        batch element ``n``.

    Returns
    -------
    w_f : torch.Tensor
        Of shape ``(lmax.max(), binom(tmax - lmax.min() + 1, kmax), nmax)``, where
        ``w_[:lmax[n], :binom(tmax - lmax[n] + 1, kmax), n]`` contain the odds relevant
        to batch index ``n``. For ``kmax == 1`` and ``t < tmax - lmax[n] + ell + 1``,
        ``w_f[ell, t, n] == w[t + ell, n]``. The values for indices
        ``t >= tmax - lmax[n] + ell + 1`` and ``ell >= lmax[n]`` are all set to `zero`.
    """
    device = w.device
    assert device == lmax.device
    assert w.dim() == 2
    if batch_first:
        w = w.t()
    tmax, nmax = w.size()
    assert lmax.dim() == 1 and lmax.size(0) == nmax
    # FIXME(anon): I believe the logic is the same regardless of kmax, but check
    lmax_min, lmax_max = int(lmax.min().item()), int(lmax.max().item())
    if not lmax_max:
        out_shape = torch.Size(
            [0, nmax, tmax + 1] if batch_first else [0, tmax + 1, nmax]
        )
        return torch.empty(out_shape, device=device, dtype=w.dtype)
    diffdim = tmax - lmax_min + 1
    padding = max(0, diffdim + lmax_max - tmax)
    if padding:
        w = torch.cat([w, torch.full((padding, nmax), fill, device=device)])
    w_f_ = []
    for ell in range(lmax_max):
        w_f_.append(w[ell : diffdim + ell].t())
    # some weirdness here. We're probably going to call R_forward on this later, and the
    # cumulative sum is more efficient on the rightmost axis, so we prefer contiguous on
    # batch_first = True (though we prefer w input to be contiguous the other way)
    w_f = torch.stack(w_f_, 0)  # .masked_fill(mask, -float("inf"))
    return w_f if batch_first else w_f.transpose(1, 2)


@torch.jit.script_if_tracing
def R_forward(
    w_f: torch.Tensor,
    lmax: torch.Tensor,
    return_all: bool = False,
    batch_first: bool = False,
) -> torch.Tensor:
    if not batch_first:
        w_f = w_f.transpose(1, 2)
    lmax = lmax.long()
    L, N, D = w_f.shape
    r = [w_f.new_ones(N, D)]
    for ell in range(L):
        r.append((w_f[ell] * r[ell]).cumsum(1))
    r = torch.stack(r)
    if return_all:
        return r if batch_first else r.transpose(1, 2)
    else:
        r = r[..., -1]
        return r.gather(0, lmax.unsqueeze(0)).squeeze(0)


@torch.jit.script_if_tracing
def log_R_forward(
    logits_f: torch.Tensor,
    lmax: torch.Tensor,
    return_all: bool = False,
    batch_first: bool = False,
    eps_ninf: float = config.EPS_NINF,
) -> torch.Tensor:
    assert logits_f.dim() == 3 and lmax.dim() == 1
    if not batch_first:
        logits_f = logits_f.transpose(1, 2)
    lmax = lmax.long()
    L, N, D = logits_f.shape
    logits_f = logits_f.clamp_min(eps_ninf)
    lr = [logits_f.new_zeros(N, D)]
    for ell in range(L):
        lr.append((logits_f[ell] + lr[ell]).logcumsumexp(1))
    lr = torch.stack(lr)
    if return_all:
        return lr if batch_first else lr.transpose(1, 2)
    else:
        lr = lr[..., -1]
        return lr.gather(0, lmax.unsqueeze(0)).squeeze(0)


# TESTS


def test_extract_relevant_odds_forward():
    tmax, nmax = 123, 45
    lmax = torch.arange(nmax)
    w = torch.arange(tmax * nmax).view(nmax, tmax).t()
    w_f = extract_relevant_odds_forward(w, lmax)
    assert w_f.size() == torch.Size([nmax - 1, tmax + 1, nmax])
    for n in range(nmax):
        w_f_n = w_f[..., n]
        # assert (w_f_n[:, tmax - lmax[n] + 1 :] == 0).all()
        # assert (w_f_n[lmax[n] :] == 0).all()
        w_f_n = w_f_n[: lmax[n], : tmax - lmax[n] + 1]
        w_f_n_exp = torch.arange(w_f_n.size(1)) + n * tmax
        for ell in range(lmax[n].item()):
            assert (w_f_n[ell] == (w_f_n_exp + ell)).all()


def test_R_forward():
    torch.manual_seed(2)
    w = torch.randn((4, 3)).exp().requires_grad_(True)
    lmax = torch.tensor([3, 1, 2])
    g = torch.randn((3, 4, 3)).exp().requires_grad_(True)
    r_exp = torch.stack(
        [
            w[0, 0]
            * g[0, 0, 0]
            * (
                w[1, 0] * g[1, 1, 0] * (w[2, 0] * g[2, 2, 0] + w[3, 0] * g[2, 3, 0])
                + w[2, 0] * g[1, 2, 0] * w[3, 0] * g[2, 3, 0]
            )
            + w[1, 0] * g[0, 1, 0] * w[2, 0] * g[1, 2, 0] * w[3, 0] * g[2, 3, 0],
            w[0, 1] * g[0, 0, 1]
            + w[1, 1] * g[0, 1, 1]
            + w[2, 1] * g[0, 2, 1]
            + w[3, 1] * g[0, 3, 1],
            w[0, 2]
            * g[0, 0, 2]
            * (w[1, 2] * g[1, 1, 2] + w[2, 2] * g[1, 2, 2] + w[3, 2] * g[1, 3, 2])
            + w[1, 2] * g[0, 1, 2] * (w[2, 2] * g[1, 2, 2] + w[3, 2] * g[1, 3, 2])
            + w[2, 2] * g[0, 2, 2] * w[3, 2] * g[1, 3, 2],
        ]
    )
    grad_w_exp, grad_g_exp = torch.autograd.grad(r_exp, [w, g], torch.ones_like(r_exp))
    g_f = torch.stack(
        [
            g[0],
            torch.cat([g[1, 1:], torch.empty_like(g[1, :1])]),
            torch.cat([g[2, 2:], torch.empty_like(g[2, :2])]),
        ]
    )
    w_f = extract_relevant_odds_forward(w, lmax) * g_f
    r_act = R_forward(w_f, lmax)
    assert r_exp.shape == r_act.shape
    assert torch.allclose(r_exp, r_act)
    grad_w_act, grad_g_act = torch.autograd.grad(r_act, [w, g], torch.ones_like(r_act))
    assert torch.allclose(grad_w_exp, grad_w_act)
    assert torch.allclose(grad_g_exp, grad_g_act)


def test_log_R_forward():
    torch.manual_seed(3)
    nmax, tmax = 5, 16
    logits = torch.randn(nmax, tmax, requires_grad=True, dtype=torch.double)
    lmax = torch.randint(0, tmax + 1, (nmax,))
    logits_f = extract_relevant_odds_forward(logits, lmax, True, config.EPS_NINF)
    lg_f = torch.randn_like(logits_f, requires_grad=True)
    lwg_f = logits_f + lg_f.masked_fill(torch.isinf(logits_f), config.EPS_NINF)
    r_act = log_R_forward(lwg_f, lmax, batch_first=True)
    logits_grad_act, lg_f_grad_act = torch.autograd.grad(r_act.mean(), [logits, lg_f])

    w, g_f = logits.exp(), lg_f.exp()
    w_f = extract_relevant_odds_forward(w, lmax, True)
    wg_f = w_f * g_f.masked_fill(w_f == 0, 0.0)
    r_exp = R_forward(wg_f, lmax, batch_first=True).log()
    assert torch.allclose(r_exp, r_act)
    logits_grad_exp, lg_f_grad_exp = torch.autograd.grad(r_exp.mean(), [logits, lg_f])
    assert torch.allclose(logits_grad_exp, logits_grad_act)
    assert torch.allclose(lg_f_grad_exp, lg_f_grad_act)
