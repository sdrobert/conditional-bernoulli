import torch
import conditional_bernoulli
import bounded_bernoulli


def to_one_hot(b_index, len_, dim=0):
    shape = list(b_index.shape)
    shape[dim] = len_ + 1
    b_onehot = torch.zeros(*shape, device=b_index.device, dtype=torch.float)
    b_onehot.scatter_(dim, (b_index + 1).clamp(0), 1.)
    _, b_onehot = b_onehot.split([1, len_])
    return b_onehot


def test_sample():
    torch.manual_seed(9302)
    T, N = 20, 100000
    w = torch.rand(T)
    w[::2] = 0.
    counts = torch.tensor([9, 10]).unsqueeze(0).expand(N, 2)

    s = to_one_hot(bounded_bernoulli.sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    ), T)
    assert torch.all(s.int().sum(0) == counts)
    props1 = s.mean(1)

    s = conditional_bernoulli.sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    assert torch.all(s.int().sum(0) == counts)
    props2 = s.mean(1)

    assert torch.allclose(props1, props2, atol=1e-2)


def test_lsample():
    torch.manual_seed(2318)
    T, N = 105, 15
    logits = torch.randn(T, N)
    logits[::2] = -float('inf')
    counts = torch.randint(1, T // 2 - 1, (N,))

    torch.manual_seed(1)
    b1 = bounded_bernoulli.sample(logits.exp(), counts)

    torch.manual_seed(1)
    b2 = bounded_bernoulli.lsample(logits, counts)

    assert torch.all((b1 >= 0).sum(0).int() == counts)
    assert torch.all((b2 >= 0).sum(0).int() == counts)

    assert torch.all(b1 == b2)


def test_probs():
    torch.manual_seed(31017)
    T, N = 7, 20
    w = torch.rand(T, N)
    counts = torch.randint(1, T + 1, (N,))
    b = bounded_bernoulli.sample(w, counts)
    w.requires_grad_(True)

    p = bounded_bernoulli.probs(w, b)
    g, = torch.autograd.grad(p, w, torch.ones_like(p))
    # every weight should have a gradient, if from the first draft alone
    assert not torch.any(torch.isclose(g, torch.tensor(0.)))

    b = to_one_hot(b, T)
    p2 = conditional_bernoulli.probs(w, b)
    assert torch.allclose(p.prod(0), p2)


def test_lprobs():
    torch.manual_seed(3284701)
    T, N = 13, 20
    logits = torch.randn(T, N)
    logits[::2] = -float('inf')
    counts = torch.randint(1, T // 2, (N,))
    b = bounded_bernoulli.lsample(logits, counts)

    logits.requires_grad_(True)
    p1 = bounded_bernoulli.lprobs(logits, b).exp()
    g1, = torch.autograd.grad(p1, logits, torch.ones_like(p1))

    logits.requires_grad_(True)
    w = logits.exp()
    p2 = bounded_bernoulli.probs(w, b)
    g2, = torch.autograd.grad(p2, logits, torch.ones_like(p2))

    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(g1, g2, atol=1e-5)
