import torch

from poisson_binomial import lR, R


# sample using the "direct" method from Chen '97. It's fast and numerically
# stable

def sample(w, counts):
    # in w = (T, *), counts = int or (*)
    # out b = (T, *)
    w = w.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=w.device)
    counts = counts.expand_as(w[0]).detach()

    T = w.shape[0]
    max_count = counts.max().item()
    assert 0 <= max_count <= T

    b = []
    Rhist = R(w, max_count, True)  # [T + 1, ]
    U = torch.rand_like(w[0]) * Rhist[-1].gather(0, counts.unsqueeze(0))[0]
    # N.B. counts is now going to double for n - r in Chen '97
    for k in range(1, T + 1):
        R_k = Rhist[-k - 1].gather(0, counts.unsqueeze(0))[0]
        match = U >= R_k
        b.insert(0, match)
        U = torch.where(match, (U - R_k) / w[T - k], U)
        counts = counts - match.long()
    del Rhist, U

    return torch.stack(b).float()


def lsample(logits, counts):
    # in logits = (T, *), counts = int or (*)
    # out b = (T, *)
    logits = logits.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=logits.device)
    counts = counts.expand_as(logits[0]).detach()

    T = logits.shape[0]
    max_count = counts.max().item()
    assert 0 <= max_count <= T

    b = []
    Rhist = lR(logits, max_count, True)  # [T + 1, ]
    U = (
        torch.rand_like(logits[0]).log() +
        Rhist[-1].gather(0, counts.unsqueeze(0))[0]
    )
    # N.B. counts is now going to double for n - r in Chen '97
    for k in range(1, T + 1):
        R = Rhist[-k - 1].gather(0, counts.unsqueeze(0))[0]
        match = U >= R
        b.insert(0, match)
        U = torch.where(
            match, U + torch.log1p(-((R - U).exp())) - logits[T - k], U)
        counts = counts - match.long()
    del Rhist, U

    return torch.stack(b).float()


def probs(w, b):
    # in w = b = (T, *)
    counts = b.sum(0).long()
    w = w.masked_fill(~b.bool(), 1.)
    num = w.prod(0)
    denom = R(w, counts.max()).gather(0, counts.unsqueeze(0))[0]
    return num / denom


def lprobs(logits, b):
    # in logits = b = (T, *)
    counts = b.sum(0).long()
    logits = logits.masked_fill(~b.bool(), 0.)
    num = logits.sum(0)
    denom = lR(logits, counts.max()).gather(0, counts.unsqueeze(0))[0]
    return num - denom
