import torch
import poisson_binomial
import conditional_bernoulli


def naive_sample(w, counts):
    # in w = (T, *), counts = int or (*)
    # out b = (T, *)
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=w.device)
    w = w.detach()
    counts = counts.expand_as(w[0]).detach()
    T = w.shape[0]
    max_count = counts.max().item()
    assert 0 <= max_count <= T
    min_count = counts.min()
    assert min_count >= 0
    with_replacement = torch.cartesian_prod(
        *(torch.eye(2, device=w.device, dtype=bool)[-1],) * T).T

    # first thing we do is get rid of the combinations that will *never* match
    match = with_replacement.sum(0)
    match = (match <= max_count) & (match >= min_count)
    with_replacement = with_replacement[:, match]
    combos = with_replacement.shape[-1]
    del match

    # get combos to last dimension
    with_replacement = with_replacement.view(
        *((T,) + (1,) * counts.dim() + (combos,)))
    with_replacement = with_replacement.expand(
        *((-1,) + counts.shape + (-1,)))

    # then only keep the lines with matching counts
    wo_replacement = with_replacement.sum(0) == counts.unsqueeze(-1)
    wo_replacement = wo_replacement.unsqueeze(0) & with_replacement
    del with_replacement

    # copy weights to those lines
    w = torch.where(
        wo_replacement, w.unsqueeze(-1), torch.tensor(1., device=w.device))

    # take the product of count weights
    w = torch.where(
        wo_replacement.any(0),
        w.prod(0),
        torch.zeros_like(w[0]),
    )

    # normalize into a categorical probability distribution and sample
    if min_count:
        # there should be at least one weight
        p = w / w.sum(-1, keepdim=True)
    else:
        # some conditional expects zero samples, meaning no weights in the
        # denominator. There's only one sample this can be, b = 0.
        norm = w.sum(-1, keepdim=True)
        zero_p = torch.ones_like(w)
        zero_p = zero_p.masked_fill(wo_replacement.any(0), 0.)
        p = torch.where(
            norm > 0.,
            w / norm,
            zero_p
        )
        del zero_p, norm

    idxs = torch.distributions.Categorical(probs=p).sample()
    del w, p

    # the categorical sample points us to the index in combos per batch
    # element. We need to convert to the bernoulli sample by pulling out
    # terms from wo_replacement
    idxs = idxs.view(1, *counts.shape, 1)
    idxs = idxs.expand(T, *(-1,) * (idxs.dim() - 1))
    b = wo_replacement.gather(-1, idxs)[..., 0].float()
    del wo_replacement

    return b


def test_naive_sample():
    torch.manual_seed(3472196)
    T, N = 10, 10000
    bern_p = torch.rand(T)
    bern_p /= bern_p.sum()
    w = bern_p / (1 - bern_p)
    counts = torch.arange(T + 1)
    poisson_probs = poisson_binomial.probs(w)
    s = naive_sample(
        w.view(T, 1, 1).expand(T, T + 1, N),
        counts.unsqueeze(-1).expand(T + 1, N)
    )
    assert torch.all(s.int().sum(0) == counts.unsqueeze(-1))

    # this is the cool bit. Multiply our Monte Carlo conditional Bernoulli
    # probability estimate with the Poisson-Binomial prior and we recover the
    # original Bernoulli probabilities
    mc_probs = (s.mean(-1) * poisson_probs).sum(-1)
    assert torch.allclose(bern_p, mc_probs, atol=1e-2)


def test_sample():
    torch.manual_seed(40272)
    T, N = 10, 100000
    w = torch.rand(T)
    w[::2] = 0.
    counts = torch.tensor([4, 5]).unsqueeze(0).expand(N, 2)

    s = naive_sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    props = s.mean(1)
    del s

    s = conditional_bernoulli.sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    assert torch.all(s.int().sum(0) == counts)
    props2 = s.mean(1)

    assert torch.allclose(props, props2, atol=1e-2)


def test_lsample():
    torch.manual_seed(5429)
    T, N = 30, 12
    logits = torch.randn(T, N)
    logits[1::2] = -float('inf')
    counts = torch.randint(1, T // 2 - 1, (N,))

    torch.manual_seed(1)
    b1 = conditional_bernoulli.sample(logits.exp(), counts)

    torch.manual_seed(1)
    b2 = conditional_bernoulli.lsample(logits, counts)

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
