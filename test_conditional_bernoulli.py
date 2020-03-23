import torch
import poisson_binomial
import conditional_bernoulli


def test_naive_sample():
    torch.manual_seed(3472196)
    T, N = 10, 10000
    bern_p = torch.rand(T)
    bern_p /= bern_p.sum()
    w = bern_p / (1 - bern_p)
    counts = torch.arange(T + 1)
    poisson_probs = poisson_binomial.probs(w)
    s = conditional_bernoulli.naive_sample(
        w.view(T, 1, 1).expand(T, T + 1, N),
        counts.unsqueeze(-1).expand(T + 1, N)
    )
    assert torch.all(s.int().sum(0) == counts.unsqueeze(-1))

    # this is the cool bit. Multiply our Monte Carlo conditional Bernoulli
    # probability estimate with the Poisson-Binomial prior and we recover the
    # original Bernoulli probabilities
    mc_probs = (s.mean(-1) * poisson_probs).sum(-1)
    assert torch.allclose(bern_p, mc_probs, atol=1e-2)


def test_draft_sample():
    torch.manual_seed(32412)
    T, N = 10, 100000
    w = torch.rand(T)
    counts = torch.tensor([3, 4]).unsqueeze(0).expand(N, 2)

    s = conditional_bernoulli.naive_sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    props = s.mean(1)
    del s

    s = conditional_bernoulli.draft_sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    assert torch.all(s.int().sum(0) == counts)
    props2 = s.mean(1)

    assert torch.allclose(props, props2, atol=1e-2)


def test_draft_lsample():
    torch.manual_seed(472043)
    T, N = 30, 12
    theta = torch.randn(T, N)
    theta[::2] = -float('inf')
    counts = torch.randint(1, T // 2, (N,))

    torch.manual_seed(1)
    b1 = conditional_bernoulli.draft_sample(theta.exp(), counts)

    torch.manual_seed(1)
    b2 = conditional_bernoulli.draft_lsample(theta, counts)

    assert torch.all(b1 == b2)


def test_direct_sample():
    torch.manual_seed(40272)
    T, N = 10, 100000
    w = torch.rand(T)
    w[::2] = 0.
    counts = torch.tensor([4, 5]).unsqueeze(0).expand(N, 2)

    s = conditional_bernoulli.naive_sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    props = s.mean(1)
    del s

    s = conditional_bernoulli.direct_sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    assert torch.all(s.int().sum(0) == counts)
    props2 = s.mean(1)

    assert torch.allclose(props, props2, atol=1e-2)


def test_direct_lsample():
    torch.manual_seed(5429)
    T, N = 30, 12
    logits = torch.randn(T, N)
    logits[1::2] = -float('inf')
    counts = torch.randint(1, T // 2 - 1, (N,))

    torch.manual_seed(1)
    b1 = conditional_bernoulli.direct_sample(logits.exp(), counts)

    torch.manual_seed(1)
    b2 = conditional_bernoulli.direct_lsample(logits, counts)

    assert torch.all(b1.sum(0).int() == counts)
    assert torch.all(b2.sum(0).int() == counts)

    assert torch.all(b1 == b2)


def test_probs():
    torch.manual_seed(234701)
    T, N = 50, 13
    w = torch.rand(T, N)
    w[::3] = 0.
    b = torch.randint(0, 2, (T, N), dtype=bool)
    b &= w != 0.
    counts = b.sum(0).long()

    pb_prob = poisson_binomial.probs(w).gather(0, counts.unsqueeze(0))[0]
    bern_prob = torch.where(b, w / (1. + w), 1 - w / (1. + w))
    prob_exp = bern_prob.prod(0) / pb_prob
    assert not torch.any(prob_exp == 0.)

    prob_act = conditional_bernoulli.probs(w, b)
    assert torch.allclose(prob_exp, prob_act)


def test_lprobs():
    torch.manual_seed(3284701)
    T, N = 102, 18
    logits = torch.randn(T, N)
    logits[::4] = -float('inf')
    b = torch.randint(0, 2, (T, N), dtype=bool)
    b &= torch.isfinite(logits)

    logits.requires_grad_(True)
    p1 = conditional_bernoulli.lprobs(logits, b).exp()
    g1, = torch.autograd.grad(p1, logits, torch.ones_like(p1))

    logits.requires_grad_(True)
    w = logits.exp()
    p2 = conditional_bernoulli.probs(w, b)
    g2, = torch.autograd.grad(p2, logits, torch.ones_like(p2))

    assert torch.allclose(p1, p2)
    assert torch.allclose(g1, g2)
