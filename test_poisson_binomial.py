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
