import poisson_binomial
import torch
import pytest


def test_naive_R():
    torch.manual_seed(10237)
    w = torch.rand(4)
    w.requires_grad_(True)
    w_1, w_2, w_3, w_4 = w.unsqueeze(-1)
    R_exp = torch.cat([
        torch.tensor([1.]),
        w_1 + w_2 + w_3 + w_4,
        w_1 * w_2 + w_1 * w_3 + w_1 * w_4 + w_2 * w_3 + w_2 * w_4 + w_3 * w_4,
        w_1 * w_2 * w_3 + w_1 * w_2 * w_4 + w_1 * w_3 * w_4 + w_2 * w_3 * w_4,
        w_1 * w_2 * w_3 * w_4,
        torch.tensor([0., 0.]),
    ], 0)
    g_exp, = torch.autograd.grad(R_exp, w, torch.ones_like(R_exp))
    w.requires_grad_(True)
    R_act = poisson_binomial.naive_R(w, 6)
    assert R_exp.shape == R_act.shape
    assert torch.allclose(R_exp, R_act, atol=1e-5)
    g_act, = torch.autograd.grad(R_act, w, torch.ones_like(R_act))
    assert torch.allclose(g_exp, g_act, atol=1e-5)


def test_shift_R():
    torch.manual_seed(2154)
    w = torch.rand(50)
    R_exp = poisson_binomial.naive_R(w, 10)
    R_act = poisson_binomial.shift_R(w, 10)
    assert torch.allclose(R_exp, R_act)
    R_act = poisson_binomial.shift_R(w.unsqueeze(-1).expand(-1, 30), 10)
    assert torch.allclose(R_exp, R_act[..., 11])


def test_shift_R_history():
    torch.manual_seed(7420)
    w = torch.rand(4)
    w1, w2, w3, w4 = w.tolist()
    Rhist_exp = torch.tensor([
        [1, 1, 1, 1, 1],
        [0, w1, w1 + w2, w1 + w2 + w3, w1 + w2 + w3 + w4],
        [
            0, 0, w1 * w2, w1 * w2 + w1 * w3 + w2 * w3, w1 * w2 +
            w1 * w3 + w1 * w4 + w2 * w3 + w2 * w4 + w3 * w4
        ]
    ]).T
    Rhist_act = poisson_binomial.shift_R(
        w.unsqueeze(-1).expand(-1, 10), 2, True)
    assert torch.allclose(Rhist_exp, Rhist_act[..., 4])


def test_R_properties():
    torch.manual_seed(4201)
    T, N, k = 30, 4, 5
    w = torch.rand(T, N)

    # sum_j w_j R(k-1, C\{j}) = k R(k, C)
    w_sub_j = w.unsqueeze(0).masked_fill(
        torch.eye(T, dtype=bool).unsqueeze(-1), 0.).transpose(0, 1)
    assert torch.allclose(w_sub_j.sum(1) / (T - 1), w)
    R_sub_j = poisson_binomial.shift_R(w_sub_j, k)
    R_k_act = (R_sub_j[-2] * w).sum(0) / k
    R = poisson_binomial.shift_R(w, k)
    R_k_exp = R[-1]
    assert torch.allclose(R_k_act, R_k_exp)

    # sum_j R(k - 1, C\{j}) = (T - k) R(k, C)
    assert torch.allclose(R_sub_j[-1].sum(0), (T - k) * R[-1])

    # sum_i R(i, C) R(k - i, C^c) = R(k, C)
    w_left = w[:T // 2]
    w_right = w[T // 2:]
    R_left = poisson_binomial.shift_R(w_left, k)
    R_right = poisson_binomial.shift_R(w_right, k)
    R_k_act = (R_left * R_right.flip(0)).sum(0)
    assert torch.allclose(R_k_act, R_k_exp)


def test_shift_R_zero_weights():
    torch.manual_seed(1702)
    T, N, k_max = 30, 10, 4
    w_lens = torch.randint(k_max, T + 1, (N,))
    w = []
    R_exp = []
    g_exp = []
    for w_len in w_lens.tolist():
        w_n = torch.rand(w_len)
        w_n[0] = 0.
        w_n.requires_grad_(True)
        R_n = poisson_binomial.shift_R(w_n, k_max)
        g_n, = torch.autograd.grad(R_n, w_n, torch.ones_like(R_n))
        R_exp.append(R_n)
        g_exp.append(g_n)
        w.append(w_n)
    w = torch.nn.utils.rnn.pad_sequence(w)
    R_exp = torch.stack(R_exp, dim=-1)
    w.requires_grad_(True)
    R_act = poisson_binomial.shift_R(w, k_max)
    assert torch.allclose(R_exp, R_act)
    g_act, = torch.autograd.grad(R_act, w, torch.ones_like(R_act))
    # zero weights can still have a gradient, but that shouldn't affect the
    # gradient of other weights
    for w_len, g_exp_n, g_act_n in zip(w_lens, g_exp, g_act.T):
        assert torch.all(torch.isfinite(g_exp_n))
        assert torch.allclose(g_exp_n[:w_len], g_act_n[:w_len])


@pytest.mark.parametrize('keep_hist', [True, False])
def test_shift_log_R(keep_hist):
    torch.manual_seed(198236)
    w = torch.rand(50, 4, 30, 10)
    R_exp = poisson_binomial.shift_R(w, 20, keep_hist)
    R_act = poisson_binomial.shift_log_R(w.log(), 20, keep_hist).exp()
    assert torch.allclose(R_exp, R_act)


def test_shift_log_R_inf_logits():
    torch.manual_seed(3291)
    T, N, k_max = 53, 5, 3
    logits_lens = torch.randint(k_max, T + 1, (N,))
    logits = []
    R_exp = []
    g_exp = []
    for logits_len in logits_lens.tolist():
        logits_n = torch.randn(logits_len)
        logits_n[0] = -float('inf')
        logits_n.requires_grad_(True)
        R_n = poisson_binomial.shift_log_R(logits_n, k_max)
        g_n, = torch.autograd.grad(R_n, logits_n, torch.ones_like(R_n))
        R_exp.append(R_n)
        g_exp.append(g_n)
        logits.append(logits_n)
    logits = torch.nn.utils.rnn.pad_sequence(
        logits, padding_value=-float('inf'))
    R_exp = torch.stack(R_exp, dim=-1)
    logits.requires_grad_(True)
    R_act = poisson_binomial.shift_log_R(logits, k_max)
    assert torch.allclose(R_exp, R_act)
    g_act, = torch.autograd.grad(R_act, logits, torch.ones_like(R_act))
    for logits_len, g_exp_n, g_act_n in zip(logits_lens, g_exp, g_act.T):
        assert not torch.any(torch.isnan(g_exp_n))
        assert torch.allclose(g_exp_n[:logits_len], g_act_n[:logits_len])
        # the gradient isn't really defined for -inf in the log-space, but it
        # works out to be the non-threatening zero here
        assert torch.all(g_act_n[logits_len:] == 0.)


def test_poisson_binomial_probs():
    torch.manual_seed(400)
    T, N = 10, 1000000
    bern_p = torch.rand(T)
    bern_p /= torch.rand(T).sum()
    w = bern_p / (1 - bern_p)
    s = torch.distributions.Bernoulli(
        probs=bern_p.unsqueeze(-1).expand(T, N)).sample()
    counts = s.sum(0)
    mc_probs = (torch.arange(T + 1).unsqueeze(-1) == counts).float().mean(-1)
    assert torch.isclose(mc_probs.sum(), torch.tensor(1.))
    pred_probs = poisson_binomial.probs(w)
    assert torch.allclose(pred_probs, mc_probs, atol=1e-3)


def test_poisson_binomial_log_probs():
    torch.manual_seed(24229027)
    T, N = 30, 1000000
    logits = torch.randn(T)
    s = torch.distributions.Bernoulli(
        logits=logits.unsqueeze(-1).expand(T, N)).sample()
    counts = s.sum(0)
    mc_probs = (torch.arange(T + 1).unsqueeze(-1) == counts).float().mean(-1)
    assert torch.isclose(mc_probs.sum(), torch.tensor(1.))
    pred_probs = poisson_binomial.lprobs(logits).exp()
    assert torch.isclose(pred_probs.sum(), torch.tensor(1.))
    assert torch.allclose(pred_probs, mc_probs, atol=1e-3)
