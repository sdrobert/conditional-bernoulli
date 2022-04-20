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


@torch.jit.script_if_tracing
def extract_relevant_odds_forward(
    w: torch.Tensor,
    given_count: torch.Tensor,
    batch_first: bool = False,
    fill: float = 0.0,
) -> torch.Tensor:
    device = w.device
    assert device == given_count.device
    assert w.dim() == 2
    if batch_first:
        w = w.t()
    T, N = w.size()
    assert given_count.dim() == 1 and given_count.size(0) == N
    # FIXME(anon): I believe the logic is the same regardless of kmax, but check
    Lmin, Lmax = int(given_count.min().item()), int(given_count.max().item())
    if not Lmax:
        out_shape = torch.Size([0, N, T + 1] if batch_first else [0, T + 1, N])
        return torch.empty(out_shape, device=device, dtype=w.dtype)
    D = T - Lmin + 1
    padding = max(0, D + Lmax - T)
    if padding:
        w = torch.cat([w, torch.full((padding, N), fill, device=device)])
    w_f_ = []
    for ell in range(Lmax):
        w_f_.append(w[ell : D + ell].t())
    # some weirdness here. We're probably going to call R_forward on this later, and the
    # cumulative sum is more efficient on the rightmost axis, so we prefer contiguous on
    # batch_first = True (though we prefer w input to be contiguous the other way)
    w_f = torch.stack(w_f_, 0)  # .masked_fill(mask, -float("inf"))
    return w_f if batch_first else w_f.transpose(1, 2)


@torch.jit.script_if_tracing
def R_forward(
    w_f: torch.Tensor,
    given_count: torch.Tensor,
    return_all: bool = False,
    batch_first: bool = False,
) -> torch.Tensor:
    if not batch_first:
        w_f = w_f.transpose(1, 2)
    given_count = given_count.long()
    L, N, D = w_f.shape
    r = [w_f.new_ones(N, D)]
    for ell in range(L):
        r.append((w_f[ell] * r[ell]).cumsum(1))
    r = torch.stack(r)
    if return_all:
        return r if batch_first else r.transpose(1, 2)
    else:
        r = r[..., -1]
        return r.gather(0, given_count.unsqueeze(0)).squeeze(0)


@torch.jit.script_if_tracing
def log_R_forward(
    logits_f: torch.Tensor,
    given_count: torch.Tensor,
    return_all: bool = False,
    batch_first: bool = False,
) -> torch.Tensor:
    assert logits_f.dim() == 3 and given_count.dim() == 1
    if not batch_first:
        logits_f = logits_f.transpose(1, 2)
    given_count = given_count.long()
    L, N, D = logits_f.shape
    lr = [logits_f.new_zeros(N, D)]
    for ell in range(L):
        lr.append((logits_f[ell] + lr[ell]).logcumsumexp(1))
    lr = torch.stack(lr)
    if return_all:
        return lr if batch_first else lr.transpose(1, 2)
    else:
        lr = lr[..., -1]
        return lr.gather(0, given_count.unsqueeze(0)).squeeze(0)


# TESTS


def test_extract_relevant_odds_forward():
    T, N = 123, 45
    lmax = torch.arange(N)
    w = torch.arange(T * N).view(N, T).t()
    w_f = extract_relevant_odds_forward(w, lmax)
    assert w_f.size() == torch.Size([N - 1, T + 1, N])
    for n in range(N):
        w_f_n = w_f[..., n]
        # assert (w_f_n[:, T - lmax[n] + 1 :] == 0).all()
        # assert (w_f_n[lmax[n] :] == 0).all()
        w_f_n = w_f_n[: lmax[n], : T - lmax[n] + 1]
        w_f_n_exp = torch.arange(w_f_n.size(1)) + n * T
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
    import pydrobert.torch.config as config

    torch.manual_seed(3)
    N, T = 5, 16
    logits = torch.randn(N, T, requires_grad=True, dtype=torch.double)
    lmax = torch.randint(0, T + 1, (N,))
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
