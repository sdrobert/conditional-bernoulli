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


def test_shift_log_R_inf_theta():
    torch.manual_seed(3291)
    T, N, k_max = 53, 5, 3
    theta_lens = torch.randint(k_max, T + 1, (N,))
    theta = []
    R_exp = []
    g_exp = []
    for theta_len in theta_lens.tolist():
        theta_n = torch.randn(theta_len).requires_grad_(True)
        R_n = poisson_binomial.shift_log_R(theta_n, k_max)
        g_n, = torch.autograd.grad(R_n, theta_n, torch.ones_like(R_n))
        R_exp.append(R_n)
        g_exp.append(g_n)
        theta.append(theta_n)
    theta = torch.nn.utils.rnn.pad_sequence(theta, padding_value=-float('inf'))
    R_exp = torch.stack(R_exp, dim=-1)
    theta.requires_grad_(True)
    R_act = poisson_binomial.shift_log_R(theta, k_max)
    assert torch.allclose(R_exp, R_act)
    g_act, = torch.autograd.grad(R_act, theta, torch.ones_like(R_act))
    # zero weights can still have a gradient, but that shouldn't affect the
    # gradient of other weights
    for theta_len, g_exp_n, g_act_n in zip(theta_lens, g_exp, g_act.T):
        assert torch.allclose(g_exp_n[:theta_len], g_act_n[:theta_len])
        assert torch.all(torch.isfinite(g_act_n[theta_len:]))
