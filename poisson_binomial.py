import torch


# this isn't numerically stable
def naive_R(w, k_max):
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
        if k_max:
            log_Rrest = torch.logsumexp(torch.stack([log_Rrest, x], 0), 0)
        else:
            log_Rrest = x
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


def naive_sample_conditional_bernoulli(w, counts):
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


def draft_sample_conditional_bernoulli(w, counts):
    # in w = (T, *), counts = int or (*)
    # out b = (T, *)
    w = w.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=w.device)
    counts = counts.expand_as(w[0]).detach()

    orig_shape = w.shape
    T = orig_shape[0]
    w = w.flatten(1)
    counts = counts.flatten()

    max_count = counts.max().item()
    assert 0 <= max_count <= T
    b = torch.zeros_like(w, dtype=bool)
    if not max_count:
        return b.view(orig_shape)

    dummy_seq = torch.ones_like(w.T)
    dummy_seq /= T

    next_sample = torch.eye(T, device=w.device, dtype=bool)

    # presented in '97 paper. I've had better luck with numerical stability
    # than the '94 one, though it's less efficient
    for k in range(1, max_count + 1):
        still_sampling = (counts >= k).unsqueeze(0)

        w = w.masked_fill(b, 0.)

        w_sub_j = w.unsqueeze(1).masked_fill(next_sample.unsqueeze(-1), 0.)

        R_sub_j = shift_R(w_sub_j, max_count - k)
        R_km1 = R_sub_j.gather(
           0,
           (counts - k).clamp(0).unsqueeze(0).expand_as(w).unsqueeze(0)
        )[0]

        pi = (w * R_km1).T.contiguous()
        norm = pi.sum(-1, keepdim=True)
        pi = torch.where(still_sampling.T, pi / norm, dummy_seq)

        last = torch.distributions.OneHotCategorical(
            probs=pi).sample().T.bool()
        b |= last & still_sampling

    return b.float().view(orig_shape)


def draft_sample_log_conditional_bernoulli(logits, counts):
    # in logits = (T, *), counts = int or (*)
    # out b = (T, *)
    logits = logits.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=logits.device)
    counts = counts.expand_as(logits[0]).detach()

    orig_shape = logits.shape
    T = orig_shape[0]
    logits = logits.flatten(1)
    counts = counts.flatten()

    max_count = counts.max().item()
    assert 0 <= max_count <= T
    b = torch.zeros_like(logits, dtype=bool)
    if not max_count:
        return b.view(orig_shape)

    dummy_seq = torch.full_like(logits.T, 0.)
    next_sample = torch.eye(T, device=logits.device, dtype=bool)
    ninf = -float('inf')

    for k in range(1, max_count + 1):
        still_sampling = (counts >= k).unsqueeze(0)

        logits = logits.masked_fill(b, ninf)

        logits_sub_j = logits.unsqueeze(1).masked_fill(
            next_sample.unsqueeze(-1), ninf)

        log_R_sub_j = shift_log_R(logits_sub_j, max_count - k)
        log_R_km1 = log_R_sub_j.gather(
           0,
           (
               (counts - k).clamp(0).unsqueeze(0)
               .expand_as(logits).unsqueeze(0)
            )
        )[0]

        log_pi = (logits + log_R_km1).T.contiguous()
        log_pi = torch.where(still_sampling.T, log_pi, dummy_seq)

        last = torch.distributions.OneHotCategorical(
            logits=log_pi).sample().T.bool()
        b |= last & still_sampling

    return b.float().view(orig_shape)
