import count_function
import torch
import pytest


@pytest.fixture(scope="module", params=sorted(count_function.METHODS))
def count(request):
    def _count(*args, **kwargs):
        kwargs["method"] = request.param
        return count_function.count(*args, **kwargs)

    return _count


@pytest.fixture(
    params=[pytest.param("log_chen94", marks=pytest.mark.xfail)]
    + sorted(m for m in count_function.LOG_METHODS if m != "log_chen94")
)
def log_count(request):
    def _log_count(*args, **kwargs):
        kwargs["method"] = request.param
        return count_function.log_count(*args, **kwargs)

    return _log_count


@pytest.mark.parametrize("batch_size", [1, 2, 30])
def test_count_values(count, device, batch_size):
    w = torch.rand((batch_size, 4), device=device)
    w.requires_grad_(True)
    w_1, w_2, w_3, w_4 = torch.unbind(w, -1)
    ones = torch.ones_like(w_1)
    zeros = torch.zeros_like(w_1)
    C_exp = torch.stack(
        [
            ones,
            w_1 + w_2 + w_3 + w_4,
            w_1 * w_2 + w_1 * w_3 + w_1 * w_4 + w_2 * w_3 + w_2 * w_4 + w_3 * w_4,
            w_1 * w_2 * w_3 + w_1 * w_2 * w_4 + w_1 * w_3 * w_4 + w_2 * w_3 * w_4,
            w_1 * w_2 * w_3 * w_4,
            zeros,
            zeros,
        ],
        1,
    )
    (g_exp,) = torch.autograd.grad(C_exp, w, torch.ones_like(C_exp))
    w.requires_grad_(True)
    C_act = count(w, 6)
    assert C_exp.shape == C_act.shape
    assert torch.allclose(C_exp, C_act, atol=1e-5)
    (g_act,) = torch.autograd.grad(C_act, w, torch.ones_like(C_act))
    assert torch.allclose(g_exp, g_act, atol=1e-5)


@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
@pytest.mark.parametrize("batch_size", [1, 2, 30])
def test_count_history(count, device, direction, batch_size):
    w = torch.rand((4, batch_size), device=device)
    w1, w2, w3, w4 = w.unbind(0)
    ones = torch.ones((batch_size,), device=device)
    zeros = torch.zeros_like(ones)
    C_backward = torch.stack(
        [
            torch.stack([ones, ones, ones, ones, ones], 0),
            torch.stack([zeros, w4, w4 + w3, w4 + w3 + w2, w4 + w3 + w2 + w1], 0),
            torch.stack(
                [
                    zeros,
                    zeros,
                    w4 * w3,
                    w4 * w3 + w4 * w2 + w3 * w2,
                    w4 * w3 + w4 * w2 + w4 * w1 + w3 * w2 + w3 * w1 + w2 * w1,
                ],
                0,
            ),
        ],
        1,
    )
    C_forward = torch.stack(
        [
            torch.stack([ones, ones, ones, ones, ones], 0),
            torch.stack([zeros, w1, w1 + w2, w1 + w2 + w3, w1 + w2 + w3 + w4], 0),
            torch.stack(
                [
                    zeros,
                    zeros,
                    w1 * w2,
                    w1 * w2 + w1 * w3 + w2 * w3,
                    w1 * w2 + w1 * w3 + w1 * w4 + w2 * w3 + w2 * w4 + w3 * w4,
                ],
                0,
            ),
        ],
        1,
    )
    if direction == "forward":
        C_exp = C_forward
    elif direction == "backward":
        C_exp = C_backward
    else:
        C_exp = torch.stack([C_forward, C_backward], 2)
    C_act = count(w, 2, include_hist=True, direction=direction, batch_first=False)
    assert torch.allclose(C_exp, C_act, atol=1e-5)


def test_count_properties(count, device):
    T, L = 30, 5
    # properties from Chen '94
    w = torch.rand(T, device=device)

    # sum_t w_t C(L-1, w\{w_t}) = L C(L, w)
    w_sub_j = w.unsqueeze(0).masked_fill(torch.eye(T, dtype=bool, device=device), 0.0)
    assert torch.allclose(w_sub_j.sum(0) / (T - 1), w)
    C_sub_j = count(w_sub_j, L)
    C_L_act = (C_sub_j[:, -2] * w).sum(0) / L
    C_exp = count(w.unsqueeze(0), L)
    C_L_exp = C_exp[:, -1]
    assert torch.allclose(C_L_act, C_L_exp)

    # sum_t C(L - 1, w \ {w_t}) = (T - L) C(L, C)
    assert torch.allclose(C_sub_j[:, -1].sum(0), (T - L) * C_L_exp)

    # sum_ell C(ell, w_{<t}) C(L - ell, w_{>=t}) = C(L, w)
    w_left = w[: T // 2]
    w_right = w[T // 2 :]
    C_left = count(w_left.unsqueeze(0), L)
    C_right = count(w_right.unsqueeze(0), L)
    C_act = (C_left * C_right.flip(1)).sum(1)
    assert torch.allclose(C_act, C_L_exp)


def test_count_zero_weights(count, device):
    T, N, L = 30, 10, 4
    w_lens = torch.randint(L, T + 1, (N,))
    w_lens[0] = T
    w_ns = []
    C_exp = []
    g_exp = []
    for w_len in w_lens.tolist():
        w_n = torch.rand(w_len, device=device)
        w_n[0] = 0.0
        w_n.requires_grad_(True)
        C_n = count(w_n.unsqueeze(0), L)
        (g_n,) = torch.autograd.grad(C_n, w_n, torch.ones_like(C_n))
        C_exp.append(C_n.detach())
        g_exp.append(g_n)
        w_ns.append(w_n)
    w = torch.nn.utils.rnn.pad_sequence(w_ns, batch_first=True)
    C_exp = torch.cat(C_exp, 0)
    C_act = count(w, L)
    assert torch.allclose(C_exp, C_act), (C_exp - C_act).abs().max()
    g_act = torch.autograd.grad(C_act, w_ns, torch.ones_like(C_act))
    for w_len, g_exp_n, g_act_n in zip(w_lens, g_exp, g_act):
        assert torch.allclose(g_exp_n, g_act_n)


@pytest.mark.parametrize("include_hist", [True, False])
@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
@pytest.mark.parametrize("batch_first", [True, False])
def test_log_count_values(log_count, device, include_hist, direction, batch_first):
    T, L, N = 10, 3, 30
    w = torch.randn((N, T) if batch_first else (T, N)).exp()
    w.requires_grad_(True)
    C_exp = count_function.count(
        w, L, include_hist=include_hist, direction=direction, batch_first=batch_first
    )
    (g_exp,) = torch.autograd.grad(C_exp, w, torch.ones_like(C_exp))
    C_act = log_count(
        w.log(),
        L,
        include_hist=include_hist,
        direction=direction,
        batch_first=batch_first,
    ).exp()
    assert torch.allclose(C_exp, C_act, rtol=1e-4)
    (g_act,) = torch.autograd.grad(C_act, w, torch.ones_like(C_act))
    assert torch.allclose(g_exp, g_act, rtol=1e-4)


def test_log_count_infinite_weights(log_count, device):
    T, N, L = 53, 5, 3
    lw_lens = torch.randint(L, T + 1, (N,))
    lw_lens[0] = T
    lw_ns = []
    lC_exp = []
    g_exp = []
    for lw_len in lw_lens.tolist():
        lw_n = torch.randn(lw_len, device=device)
        lw_n[0] = 0.0
        lw_n.requires_grad_(True)
        lC_n = log_count(lw_n.unsqueeze(0), L)
        (g_n,) = torch.autograd.grad(lC_n, lw_n, torch.ones_like(lC_n))
        lC_exp.append(lC_n.detach())
        g_exp.append(g_n)
        lw_ns.append(lw_n)
    lw = torch.nn.utils.rnn.pad_sequence(
        lw_ns, batch_first=True, padding_value=-float("inf")
    )
    lC_exp = torch.cat(lC_exp, 0)
    lC_act = log_count(lw, L)
    assert torch.allclose(lC_exp, lC_act), (lC_exp - lC_act).abs().max()
    g_act = torch.autograd.grad(lC_act, lw_ns, torch.ones_like(lC_act))
    for lw_len, g_exp_n, g_act_n in zip(lw_lens, g_exp, g_act):
        assert torch.allclose(g_exp_n, g_act_n)
