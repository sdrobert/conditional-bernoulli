import torch


# this isn't numerically stable
def naive_R(w, k_max):
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


def shift_R(w, k_max):
    # in w = (T, *)
    # out R = (k_max + 1, *)
    R0 = torch.ones_like(w[:1, ...])
    Rrest = torch.zeros((k_max,) + w[0].size(), device=w.device)
    w = w.unsqueeze(1)
    for T in range(w.shape[0]):
        Rrest = Rrest + w[T] * torch.cat([R0, Rrest[:-1]], 0)
    return torch.cat([R0, Rrest])


def shift_log_R(logits, k_max):
    # in logits = (T, *)
    # out log_R = (k_max + 1, *)
    log_R0 = torch.zeros_like(logits[:1, ...])
    log_Rrest = torch.full(
        (k_max,) + logits[0].size(), -float('inf'), device=logits.device)
    logits = logits.unsqueeze(1)
    for T in range(logits.shape[0]):
        x = torch.cat([log_R0, log_Rrest[:-1]], 0)
        x = torch.where(torch.isfinite(x), x + logits[T], x)
        log_Rrest = torch.logsumexp(torch.stack([log_Rrest, x], 0), 0)
        del x
    return torch.cat([log_R0, log_Rrest])


def probs(w):
    # in w = (T, *)
    # out p = (k_max + 1, *)
    return shift_R(w, len(w)) / (1 + w).prod(0, keepdim=True)


def lprobs(logits):
    # in logits = (T, *)
    # out log_p = (k_max + 1, *)
    return (
        shift_log_R(logits, len(logits)) +
        torch.nn.functional.logsigmoid(-logits).sum(0, keepdim=True)
    )


def sample_conditional_bernoulli_naive(w, counts):
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=w.device)
    w = w.detach()
    counts = counts.expand_as(w[0]).detach()
    T = w.shape[0]
    max_count = counts.max()
    assert T >= max_count
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


# def sample_conditional_bernoulli_draft(w, counts):
#     # in w = (T, *), counts = int or (*)
#     # out b = (T, *)
#     if not torch.is_tensor(counts):
#         counts = torch.tensor(counts, device=w.device)
#     counts = counts.expand_as(w[0]).detach()
#     max_count = counts.max.item()
#     assert max_count <= len(w)
#     b = torch.full_like(w, False)
#     for k in range(1, max_count + 1):
#         going = (counts >= k)


