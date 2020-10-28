import poisson_binomial
import torch
import pytest


@pytest.fixture(scope="module", params=["R", "R2", "R3", "R4"])
def R(request):
    if request.param == "R":
        return poisson_binomial.R
    elif request.param in {"R2", "R3", "R4"}:

        def _R(w, *args, **kwargs):
            T, star = w.shape[0], w.shape[1:]
            w = w.reshape(T, -1).transpose(0, 1)
            if request.param == "R2":
                v = poisson_binomial.R2(w, *args, **kwargs)
            elif request.param == "R3":
                v = poisson_binomial.R3(w, *args, **kwargs)
            elif request.param == "R4":
                v = poisson_binomial.R4(w, *args, **kwargs)
            v = v.transpose(0, -1)
            v = v.reshape(*(v.shape[:-1] + star))
            return v

        return _R


@pytest.fixture(scope="module", params=["lR", "lR4"])
def lR(request):
    if request.param == "lR":
        return poisson_binomial.lR
    elif request.param in {"lR4"}:

        def _lR(logits, *args, **kwargs):
            T, star = logits.shape[0], logits.shape[1:]
            logits = logits.reshape(T, -1).transpose(0, 1)
            if request.param == "lR4":
                v = poisson_binomial.lR4(logits, *args, **kwargs)
            v = v.transpose(0, -1)
            v = v.reshape(*(v.shape[:-1] + star))
            return v

        return _lR


# this isn't numerically stable
def naive_R(w, k_max):
    # in w = (S,)
    (S,) = w.shape
    assert 0 <= k_max
    R = torch.ones(1, device=w.device)
    T = (
        (
            ((-w).unsqueeze(0) ** torch.arange(1, k_max + 2).unsqueeze(1))
            / w.unsqueeze(0)
        )
        .sum(1)[1:]
        .flip(0)
    )  # [-1^{k+1} T(k), -1^k T(k - 1), ..., T(1)]
    for k in range(1, k_max + 1):
        if k <= S:
            R_new = (R * T[-k:]).sum(0, keepdim=True) / k
        else:
            R_new = torch.zeros(1)
        R = torch.cat([R, R_new], dim=0)
    return R


def test_naive_R():
    torch.manual_seed(10237)
    w = torch.rand(4)
    w.requires_grad_(True)
    w_1, w_2, w_3, w_4 = w.unsqueeze(-1)
    R_exp = torch.cat(
        [
            torch.tensor([1.0]),
            w_1 + w_2 + w_3 + w_4,
            w_1 * w_2 + w_1 * w_3 + w_1 * w_4 + w_2 * w_3 + w_2 * w_4 + w_3 * w_4,
            w_1 * w_2 * w_3 + w_1 * w_2 * w_4 + w_1 * w_3 * w_4 + w_2 * w_3 * w_4,
            w_1 * w_2 * w_3 * w_4,
            torch.tensor([0.0, 0.0]),
        ],
        0,
    )
    (g_exp,) = torch.autograd.grad(R_exp, w, torch.ones_like(R_exp))
    w.requires_grad_(True)
    R_act = naive_R(w, 6)
    assert R_exp.shape == R_act.shape
    assert torch.allclose(R_exp, R_act, atol=1e-5)
    (g_act,) = torch.autograd.grad(R_act, w, torch.ones_like(R_act))
    assert torch.allclose(g_exp, g_act, atol=1e-5)


@pytest.mark.parametrize("reverse", [True, False])
def test_R(reverse, R):
    torch.manual_seed(2154)
    w = torch.rand(50)
    R_exp = naive_R(w, 10)
    R_act = R(w, 10, reverse=reverse)
    assert torch.allclose(R_exp, R_act)
    R_act = R(w.unsqueeze(-1).expand(-1, 30), 10)
    assert torch.allclose(R_exp.unsqueeze(-1).expand_as(R_act), R_act), (
        R_exp[:5],
        R_act[:3, :5],
    )


@pytest.mark.parametrize("reverse", [True, False])
def test_R_history(reverse, R):
    torch.manual_seed(7420)
    w = torch.rand(4)
    w1, w2, w3, w4 = w.tolist()
    if reverse:
        Rhist_exp = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, w4, w4 + w3, w4 + w3 + w2, w4 + w3 + w2 + w1],
                [
                    0,
                    0,
                    w4 * w3,
                    w4 * w3 + w4 * w2 + w3 * w2,
                    w4 * w3 + w4 * w2 + w4 * w1 + w3 * w2 + w3 * w1 + w2 * w1,
                ],
            ]
        ).T
    else:
        Rhist_exp = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, w1, w1 + w2, w1 + w2 + w3, w1 + w2 + w3 + w4],
                [
                    0,
                    0,
                    w1 * w2,
                    w1 * w2 + w1 * w3 + w2 * w3,
                    w1 * w2 + w1 * w3 + w1 * w4 + w2 * w3 + w2 * w4 + w3 * w4,
                ],
            ]
        ).T
    Rhist_act = R(w.unsqueeze(-1).expand(-1, 2), 2, True, reverse)
    assert torch.allclose(Rhist_exp, Rhist_act[..., 1])


def test_R_properties(R):
    torch.manual_seed(4201)
    T, N, k = 30, 4, 5
    w = torch.rand(T, N)

    # sum_j w_j R(k-1, C\{j}) = k R(k, C)
    w_sub_j = (
        w.unsqueeze(0)
        .masked_fill(torch.eye(T, dtype=bool).unsqueeze(-1), 0.0)
        .transpose(0, 1)
    )
    assert torch.allclose(w_sub_j.sum(1) / (T - 1), w)
    R_sub_j = R(w_sub_j, k)
    R_k_act = (R_sub_j[-2] * w).sum(0) / k
    Rk = R(w, k)
    R_k_exp = Rk[-1]
    assert torch.allclose(R_k_act, R_k_exp)

    # sum_j R(k - 1, C\{j}) = (T - k) R(k, C)
    assert torch.allclose(R_sub_j[-1].sum(0), (T - k) * Rk[-1])

    # sum_i R(i, C) R(k - i, C^c) = R(k, C)
    w_left = w[: T // 2]
    w_right = w[T // 2 :]
    R_left = R(w_left, k)
    R_right = R(w_right, k)
    R_k_act = (R_left * R_right.flip(0)).sum(0)
    assert torch.allclose(R_k_act, R_k_exp)


@pytest.mark.parametrize("reverse", [True, False])
def test_R_zero_weights(reverse, R):
    torch.manual_seed(1702)
    T, N, k_max = 30, 10, 4
    w_lens = torch.randint(k_max, T + 1, (N,))
    w = []
    R_exp = []
    g_exp = []
    for w_len in w_lens.tolist():
        w_n = torch.rand(w_len)
        w_n[0] = 0.0
        w_n.requires_grad_(True)
        R_n = R(w_n, k_max, reverse=reverse)
        (g_n,) = torch.autograd.grad(R_n, w_n, torch.ones_like(R_n))
        R_exp.append(R_n)
        g_exp.append(g_n)
        w.append(w_n)
    w = torch.nn.utils.rnn.pad_sequence(w)
    R_exp = torch.stack(R_exp, dim=-1)
    w.requires_grad_(True)
    R_act = R(w, k_max, reverse=reverse)
    assert torch.allclose(R_exp, R_act)
    (g_act,) = torch.autograd.grad(R_act, w, torch.ones_like(R_act))
    # zero weights can still have a gradient, but that shouldn't affect the
    # gradient of other weights
    for w_len, g_exp_n, g_act_n in zip(w_lens, g_exp, g_act.T):
        assert torch.all(torch.isfinite(g_exp_n))
        assert torch.allclose(g_exp_n[:w_len], g_act_n[:w_len])


@pytest.mark.parametrize("keep_hist", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_log_R(keep_hist, reverse, lR):
    torch.manual_seed(198236)
    T, L, N1, N2 = 10, 3, 20, 5
    w = torch.randn(T, N1, N2).exp()
    w.requires_grad_(True)
    R_exp = poisson_binomial.R(w, L, keep_hist, reverse)
    (g_exp,) = torch.autograd.grad(R_exp, w, torch.ones_like(R_exp))
    w.requires_grad_(True)
    R_act = lR(w.log(), L, keep_hist, reverse).exp()
    assert torch.allclose(R_exp, R_act, rtol=1e-4)
    (g_act,) = torch.autograd.grad(R_act, w, (R_act != 0.0).float())
    assert torch.allclose(g_exp, g_act, rtol=1e-4)


@pytest.mark.parametrize("reverse", [True, False])
def test_log_R_inf_logits(reverse, lR):
    torch.manual_seed(3291)
    T, N, k_max = 53, 5, 3
    logits_lens = torch.randint(k_max, T + 1, (N,))
    logits = []
    R_exp = []
    g_exp = []
    for logits_len in logits_lens.tolist():
        logits_n = torch.randn(logits_len)
        logits_n[0] = -float("inf")
        logits_n.requires_grad_(True)
        R_n = lR(logits_n, k_max, reverse=reverse)
        (g_n,) = torch.autograd.grad(R_n, logits_n, torch.ones_like(R_n))
        R_exp.append(R_n)
        g_exp.append(g_n)
        logits.append(logits_n)
    logits = torch.nn.utils.rnn.pad_sequence(logits, padding_value=-float("inf"))
    R_exp = torch.stack(R_exp, dim=-1)
    logits.requires_grad_(True)
    R_act = lR(logits, k_max, reverse=reverse)
    assert torch.allclose(R_exp, R_act)
    (g_act,) = torch.autograd.grad(R_act, logits, torch.ones_like(R_act))
    for logits_len, g_exp_n, g_act_n in zip(logits_lens, g_exp, g_act.T):
        assert not torch.any(torch.isnan(g_exp_n))
        assert torch.allclose(g_exp_n[:logits_len], g_act_n[:logits_len])
        # the gradient isn't really defined for -inf in the log-space, but it
        # works out to be the non-threatening zero here
        assert torch.all(g_act_n[logits_len:] == 0.0)


def test_poisson_binomial_probs():
    torch.manual_seed(400)
    T, N = 10, 1000000
    bern_p = torch.rand(T)
    bern_p /= torch.rand(T).sum()
    w = bern_p / (1 - bern_p)
    s = torch.distributions.Bernoulli(probs=bern_p.unsqueeze(-1).expand(T, N)).sample()
    counts = s.sum(0)
    mc_probs = (torch.arange(T + 1).unsqueeze(-1) == counts).float().mean(-1)
    assert torch.isclose(mc_probs.sum(), torch.tensor(1.0))
    pred_probs = poisson_binomial.probs(w)
    assert torch.allclose(pred_probs, mc_probs, atol=1e-3)


def test_poisson_binomial_log_probs():
    torch.manual_seed(24229027)
    T, N = 30, 1000000
    logits = torch.randn(T)
    s = torch.distributions.Bernoulli(logits=logits.unsqueeze(-1).expand(T, N)).sample()
    counts = s.sum(0)
    mc_probs = (torch.arange(T + 1).unsqueeze(-1) == counts).float().mean(-1)
    assert torch.isclose(mc_probs.sum(), torch.tensor(1.0))
    pred_probs = poisson_binomial.lprobs(logits).exp()
    assert torch.isclose(pred_probs.sum(), torch.tensor(1.0))
    assert torch.allclose(pred_probs, mc_probs, atol=1e-3)
