import torch
import pytest
import distributions

# Note: many of the non-log forms of various functions within distributions are
# implemented here. We don't code 'em nicely as they quickly become unstable
# for large T, k. Nonetheless, they're easier to test.


# this isn't numerically stable
def naive_C(w, k_max):
    # in w = (S,)
    S, = w.shape
    assert 0 <= k_max
    R = torch.ones(1, device=w.device)
    T = (
        ((-w).unsqueeze(0) ** torch.arange(1, k_max + 2).unsqueeze(1))
        / w.unsqueeze(0)
    ).sum(1)[1:].flip(0)  # [-1^{k+1} T(k), -1^k T(k - 1), ..., T(1)]
    for k in range(1, k_max + 1):
        if k <= S:
            R_new = (R * T[-k:]).sum(0, keepdim=True) / k
        else:
            R_new = torch.zeros(1)
        R = torch.cat([R, R_new], dim=0)
    return R


def test_naive_C():
    torch.manual_seed(10237)
    w = torch.rand(4)
    w.requires_grad_(True)
    w_1, w_2, w_3, w_4 = w.unsqueeze(-1)
    C_exp = torch.cat([
        torch.tensor([1.]),
        w_1 + w_2 + w_3 + w_4,
        w_1 * w_2 + w_1 * w_3 + w_1 * w_4 + w_2 * w_3 + w_2 * w_4 + w_3 * w_4,
        w_1 * w_2 * w_3 + w_1 * w_2 * w_4 + w_1 * w_3 * w_4 + w_2 * w_3 * w_4,
        w_1 * w_2 * w_3 * w_4,
        torch.tensor([0., 0.]),
    ], 0)
    g_exp, = torch.autograd.grad(C_exp, w, torch.ones_like(C_exp))
    w.requires_grad_(True)
    C_act = naive_C(w, 6)
    assert C_exp.shape == C_act.shape
    assert torch.allclose(C_exp, C_act, atol=1e-5)
    g_act, = torch.autograd.grad(C_act, w, torch.ones_like(C_act))
    assert torch.allclose(g_exp, g_act, atol=1e-5)


def direct_C(w, k_max, keep_hist=False, reverse=False):
    # in w = (T, *)
    # out C = (k_max + 1, *) if keep_hist is False, otherwise
    # C = (T + 1, k_max + 1, *)
    C_first = torch.ones_like(w[:1, ...])
    C_rest = torch.zeros((k_max,) + w.shape[1:], device=w.device)
    w = w.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([C_first, C_rest])]
    T = w.shape[0]
    for t in range(T):
        if reverse:
            t = T - t - 1
        C_rest = C_rest + w[t] * torch.cat([C_first, C_rest[:-1]], 0)
        if keep_hist:
            hist.append(torch.cat([C_first, C_rest]))
    if keep_hist:
        return torch.stack(hist, 0)
    else:
        return torch.cat([C_first, C_rest])


@pytest.mark.parametrize('reverse', [True, False])
def test_direct_C(reverse):
    torch.manual_seed(2154)
    w = torch.rand(50)
    C_exp = naive_C(w, 10)
    C_act = direct_C(w, 10, reverse=reverse)
    assert torch.allclose(C_exp, C_act)
    C_act = direct_C(w.unsqueeze(-1).expand(-1, 30), 10)
    assert torch.allclose(C_exp, C_act[..., 11])


@pytest.mark.parametrize('reverse', [True, False])
def test_direct_C_history(reverse):
    torch.manual_seed(7420)
    w = torch.rand(4)
    w1, w2, w3, w4 = w.tolist()
    if reverse:
        C_exp = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, w4, w4 + w3, w4 + w3 + w2, w4 + w3 + w2 + w1],
            [
                0, 0, w4 * w3, w4 * w3 + w4 * w2 + w3 * w2, w4 * w3 +
                w4 * w2 + w4 * w1 + w3 * w2 + w3 * w1 + w2 * w1
            ]
        ]).T
    else:
        C_exp = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, w1, w1 + w2, w1 + w2 + w3, w1 + w2 + w3 + w4],
            [
                0, 0, w1 * w2, w1 * w2 + w1 * w3 + w2 * w3, w1 * w2 +
                w1 * w3 + w1 * w4 + w2 * w3 + w2 * w4 + w3 * w4
            ]
        ]).T
    C_act = direct_C(
        w.unsqueeze(-1).expand(-1, 10), 2, True, reverse)
    assert torch.allclose(C_exp, C_act[..., 4])


def test_C_properties():
    torch.manual_seed(4201)
    T, N, k = 30, 4, 5
    w = torch.rand(T, N)

    # sum_j w_j C(k-1, S\{j}) = k C(k, S)
    w_sub_j = w.unsqueeze(0).masked_fill(
        torch.eye(T, dtype=bool).unsqueeze(-1), 0.).transpose(0, 1)
    assert torch.allclose(w_sub_j.sum(1) / (T - 1), w)
    C_sub_j = direct_C(w_sub_j, k)
    C_k_act = (C_sub_j[-2] * w).sum(0) / k
    C = direct_C(w, k)
    C_k_exp = C[-1]
    assert torch.allclose(C_k_act, C_k_exp)

    # sum_j C(k - 1, S\{j}) = (T - k) C(k, S)
    assert torch.allclose(C_sub_j[-1].sum(0), (T - k) * C[-1])

    # sum_i C(i, S) C(k - i, S^c) = C(k, S)
    w_left = w[:T // 2]
    w_right = w[T // 2:]
    C_left = direct_C(w_left, k)
    C_right = direct_C(w_right, k)
    C_k_act = (C_left * C_right.flip(0)).sum(0)
    assert torch.allclose(C_k_act, C_k_exp)


@pytest.mark.parametrize('intermediate', [None, "forward", "reverse"])
@pytest.mark.parametrize('T,k_max', [[1, 1], [0, 1], [1, 0], [30, 12]])
def test_generalized_binomial_coefficient(T, k_max, intermediate):
    torch.manual_seed(198236)
    N = 200
    logits = torch.randn(T, N)
    w = logits.exp()
    C_exp = direct_C(w, k_max, intermediate, intermediate == 'reverse')
    C_act = distributions.generalized_binomial_coefficient(
        logits, k_max, intermediate).exp()
    assert torch.allclose(C_exp, C_act)


def test_generalized_binomial_coefficient_inf_logits():
    torch.manual_seed(3291)
    T, N, k_max = 53, 5, 3
    logits_lens = torch.randint(k_max, T + 1, (N,))
    logits = []
    C_exp = []
    g_exp = []
    for logits_len in logits_lens.tolist():
        logits_n = torch.randn(logits_len)
        logits_n[0] = -float('inf')
        logits_n.requires_grad_(True)
        C_n = distributions.generalized_binomial_coefficient(logits_n, k_max)
        g_n, = torch.autograd.grad(C_n, logits_n, torch.ones_like(C_n))
        C_exp.append(C_n)
        g_exp.append(g_n)
        logits.append(logits_n)
    logits = torch.nn.utils.rnn.pad_sequence(
        logits, padding_value=-float('inf'))
    C_exp = torch.stack(C_exp, dim=-1)
    logits.requires_grad_(True)
    C_act = distributions.generalized_binomial_coefficient(logits, k_max)
    assert torch.allclose(C_exp, C_act)
    g_act, = torch.autograd.grad(C_act, logits, torch.ones_like(C_act))
    for logits_len, g_exp_n, g_act_n in zip(logits_lens, g_exp, g_act.T):
        assert not torch.any(torch.isnan(g_exp_n))
        assert torch.allclose(g_exp_n[:logits_len], g_act_n[:logits_len])
        # the gradient isn't really defined for -inf in the log-space, but it
        # works out to be the non-threatening zero here
        assert torch.all(g_act_n[logits_len:] == 0.)


def test_poisson_binomial():
    # special case when all odds are equal: PB equals B
    torch.manual_seed(5216)
    M, N, T = 1000000, 10, 5
    p = torch.rand(N,).repeat(M)  # (N * M,)

    b_b = torch.distributions.Binomial(T, probs=p).sample().view(N, M)
    mean_b = b_b.mean(1)  # (N,)

    p = p.expand(T, N * M)
    b_pb = distributions.poisson_binomial(p.T).view(N, M)
    mean_pb = b_pb.float().mean(1)  # (N,)

    assert torch.allclose(mean_b, mean_pb, atol=1e-2)


def test_poisson_binomial_distribution():
    torch.manual_seed(310392)
    M, N, T = 1000000, 20, 10
    logits = torch.randn(N, T, requires_grad=True) * 2
    total_count = torch.randint(0, T + 1, (N,))

    dist = distributions.PoissonBinomial(
        total_count, logits=logits, validate_args=True)
    b = dist.sample((M,))
    assert b.shape == torch.Size((M, N))
    assert (b <= total_count.unsqueeze(0)).all()

    counts = torch.arange(T + 1)
    sample_probs = (counts.view(T + 1, 1, 1) == b).float().mean(1)
    assert sample_probs.shape == torch.Size((T + 1, N))

    dist._validate_args = False
    dist_lprobs = dist.log_prob(counts.unsqueeze(-1).float())
    assert dist_lprobs.shape == torch.Size((T + 1, N))

    assert torch.allclose(sample_probs, dist_lprobs.exp(), atol=1e-3)

    g, = torch.autograd.grad(dist_lprobs, logits, torch.ones_like(dist_lprobs))
    padding_mask = total_count.unsqueeze(-1) <= counts[:-1]
    assert torch.all(g.masked_select(padding_mask).eq(0.))
    assert not torch.any(g.masked_select(~padding_mask).eq(0.))
