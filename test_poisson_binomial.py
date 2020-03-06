import poisson_binomial
import torch


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


def test_shift_R_zero_weights():
    torch.manual_seed(1702)
    T, N, k_max = 30, 10, 4
    w_lens = torch.randint(k_max, T + 1, (N,))
    w = []
    R_exp = []
    g_exp = []
    for w_len in w_lens.tolist():
        w_n = torch.rand(w_len).requires_grad_(True)
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
        assert torch.allclose(g_exp_n[:w_len], g_act_n[:w_len])


def test_shift_log_R():
    torch.manual_seed(198236)
    w = torch.rand(50, 4, 30, 10)
    R_exp = poisson_binomial.shift_R(w, 20)
    R_act = poisson_binomial.shift_log_R(w.log(), 20).exp()
    assert torch.allclose(R_exp, R_act)


def test_shift_log_R_inf_logits():
    torch.manual_seed(3291)
    T, N, k_max = 53, 5, 3
    logits_lens = torch.randint(k_max, T + 1, (N,))
    logits = []
    R_exp = []
    g_exp = []
    for logits_len in logits_lens.tolist():
        logits_n = torch.randn(logits_len).requires_grad_(True)
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
    # zero weights can still have a gradient, but that shouldn't affect the
    # gradient of other weights
    for logits_len, g_exp_n, g_act_n in zip(logits_lens, g_exp, g_act.T):
        assert torch.allclose(g_exp_n[:logits_len], g_act_n[:logits_len])
        assert torch.all(torch.isfinite(g_act_n[logits_len:]))


def test_poisson_binomial_probs():
    torch.manual_seed(400)
    T, N = 10, 1000000
    bern_p = torch.rand(T)
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
    logits = torch.rand(T)
    s = torch.distributions.Bernoulli(
        logits=logits.unsqueeze(-1).expand(T, N)).sample()
    counts = s.sum(0)
    mc_probs = (torch.arange(T + 1).unsqueeze(-1) == counts).float().mean(-1)
    assert torch.isclose(mc_probs.sum(), torch.tensor(1.))
    pred_probs = poisson_binomial.lprobs(logits).exp()
    assert torch.isclose(pred_probs.sum(), torch.tensor(1.))
    assert torch.allclose(pred_probs, mc_probs, atol=1e-3)
