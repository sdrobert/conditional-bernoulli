import torch

from poisson_binomial import shift_log_R, shift_R


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


def draft_sample(w, counts):
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


def draft_lsample(logits, counts):
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


# altdraft sampling should yield the same results as draft sampling, but using
# a faster computation. They are based on Chen '94 and Chapter 5 of Tille's
# "Sampling Algorithms". While at this point I'm fairly certain of correctness,
# they are numerically unstable, which is why the asserts are there


def altdraft_sample(w, counts):
    # in w = (T, *), counts = int or (*)
    # out b = (T, *)
    assert False, "Numerically unstable"
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
    b = torch.zeros_like(w.T, dtype=bool)
    if not max_count:
        return b.T.view(orig_shape)

    dummy_val = 1 / T
    dummy_seq = torch.full_like(w.T, dummy_val)

    next_sample = torch.eye(T, device=w.device, dtype=bool)

    # presented in the '94 paper. Should be more efficient, but less
    # numerically stable

    # first sample is the same as before
    still_sampling = (counts >= 1).unsqueeze(-1)
    w_sub_j = w.unsqueeze(1).masked_fill(next_sample.unsqueeze(-1), 0.)

    R_sub_j = shift_R(w_sub_j, max_count - 1)
    R_km1 = R_sub_j.gather(
        0,
        (counts - 1).clamp(0).unsqueeze(0).expand_as(w).unsqueeze(0)
    )[0]

    pi = (w * R_km1).T.contiguous()
    norm = pi.sum(-1, keepdim=True)
    pi = torch.where(still_sampling, pi / norm, dummy_seq)

    last = torch.distributions.OneHotCategorical(probs=pi).sample().bool()
    b |= last & still_sampling

    # until we're done, we operate on tensors with the row dim being the
    # sequence dimension
    w = w.T
    del w_sub_j, R_sub_j, R_km1, norm

    # remaining samples are defined relative to pi
    for k in range(2, max_count + 1):
        still_sampling = (counts >= k).unsqueeze(-1)  # (N, 1)

        # tille 5.6.1

        w_last = w[last].unsqueeze(-1)  # (N, 1)
        # assert torch.all(w_last.ne(0.))
        pi_last = pi[last].unsqueeze(-1)  # (N, 1)
        # assert torch.all(pi_last.ne(0.))
        c_m_k = (counts - k + 1).clamp(1).unsqueeze(-1).float()  # (N, 1)
        num_minu = w_last * pi  # (N, T)
        num_subtra = pi_last * w  # (N, T)
        num = num_minu - num_subtra  # (N, T)
        denom = c_m_k * (w_last - w) * pi_last  # (N, T)
        match = num_minu == num_subtra
        assert torch.all(match | ((num < 0) == (denom < 0))), k

        pi = torch.where(match, torch.zeros_like(c_m_k), num / denom)  # (N, T)
        assert torch.all(pi > -1e-5), pi
        pi = pi.clamp(0.)
        match_prob = (-pi.sum(-1).clamp_max(1.) + 1.) / match.float().sum(-1)
        assert torch.all(match_prob >= 0.), match_prob
        pi = torch.where(match, match_prob.unsqueeze(-1), pi)
        pi = pi.masked_fill(b, 0)
        pi = pi.masked_fill(~still_sampling, dummy_val)

        # now that we have a probability distribution over the next sample,
        # we can clear the weight for 'last'
        w = w.masked_fill(last, 0.)

        # w_sub_j = w.T.unsqueeze(1).masked_fill(next_sample.unsqueeze(-1), 0.)

        # R_sub_j = shift_R(w_sub_j, max_count - k)
        # R_km1 = R_sub_j.gather(
        #    0,
        #    (counts - k).clamp(0).unsqueeze(0).expand_as(w.T).unsqueeze(0)
        # )[0]

        # pi2 = (w * R_km1.T)
        # norm = pi2.sum(-1, keepdim=True)
        # pi2 = torch.where(still_sampling, pi2 / norm, dummy_seq)

        # print(pi[-1], pi2[-1], match[-1])
        # assert torch.allclose(pi, pi2, atol=1e-4), k

        last = torch.distributions.OneHotCategorical(probs=pi).sample().bool()
        b |= last & still_sampling

    return b.T.float().view(orig_shape)


def altdraft_lsample(logits, counts):
    # in logits = (T, *), counts = int or (*)
    # out b = (T, *)
    assert False, "Numerically unstable"
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
    b = torch.zeros_like(logits.T, dtype=bool)
    if not max_count:
        return b.T.view(orig_shape)

    ninf = torch.tensor(-float('inf'), device=logits.device)

    next_sample = torch.eye(T, device=logits.device, dtype=bool)

    # presented in the '94 paper. Should be more efficient, but less
    # numerically stable

    # first sample is the same as before
    still_sampling = (counts >= 1).unsqueeze(-1)
    logits_sub_j = logits.unsqueeze(1).masked_fill(
        next_sample.unsqueeze(-1), ninf)

    log_R_sub_j = shift_log_R(logits_sub_j, max_count - 1)
    log_R_km1 = log_R_sub_j.gather(
        0,
        (counts - 1).clamp(0).unsqueeze(0).expand_as(logits).unsqueeze(0)
    )[0]

    logp = (logits + log_R_km1).T.contiguous()
    logp = logp.masked_fill(~still_sampling, 0.)
    logp = logp - logp.logsumexp(-1, keepdim=True)

    last = torch.distributions.OneHotCategorical(logits=logp).sample().bool()
    b |= last & still_sampling

    logits = logits.T
    del logits_sub_j, log_R_sub_j, log_R_km1

    # remaining samples are defined relative to logp
    for k in range(2, max_count + 1):
        still_sampling = (counts >= k).unsqueeze(-1)  # (N, 1)

        logits_last = logits[last].unsqueeze(-1)  # (N, 1)
        logp_last = logp[last].unsqueeze(-1)  # (N, 1)
        log_c_m_k = (counts - k + 1).clamp(1).unsqueeze(-1).float().log()

        num_minu = logits_last + logp  # (N, T)
        num_subtra = logp_last + logits  # (N, T)
        match = num_minu == num_subtra
        swap = ~match & (num_minu < num_subtra)
        assert torch.all(swap == ((logits_last < logits) & ~match)), k

        num = torch.where(
            swap,
            num_subtra + torch.log1p(-((num_minu - num_subtra).exp())),
            num_minu + torch.log1p(-((num_subtra - num_minu).exp())),
        )
        del num_minu, num_subtra

        denom = torch.where(
            swap,
            logits + torch.log1p(-((logits_last - logits).exp())),
            logits_last + torch.log1p(-((logits - logits_last).exp())),
        )
        denom = log_c_m_k + denom + logp_last

        logp = torch.where(match, ninf.view(1, 1).expand_as(num), num - denom)
        match_logp = (
            torch.log1p(-logp.logsumexp(-1)) - match.float().sum(-1).log())
        logp = torch.where(match, match_logp.unsqueeze(-1), logp)
        logp = logp.masked_fill(b, ninf)
        logp = logp.masked_fill(~still_sampling, 0.)

        logits = logits.masked_fill(b, ninf)

        # logits_sub_j = logits.T.unsqueeze(1).masked_fill(
        #     next_sample.unsqueeze(-1), ninf)

        # log_R_sub_j = shift_log_R(logits_sub_j, max_count - k)
        # log_R_km1 = log_R_sub_j.gather(
        #    0,
        #    (
        #        (counts - k).clamp(0).unsqueeze(0)
        #        .expand_as(logits.T).unsqueeze(0)
        #     )
        # )[0]

        # logp2 = logits + log_R_km1.T
        # logp2 = torch.where(
        #     torch.isinf(logp2),
        #     ninf, logp2 - logp2.logsumexp(-1, keepdim=True))
        # logp2 = logp2.masked_fill(~still_sampling, 0.)

        # assert torch.allclose(logp, logp2, atol=1e-2), k

        last = torch.distributions.OneHotCategorical(
            logits=logp).sample().bool()
        b |= last & still_sampling

    return b.T.float().view(orig_shape)


def direct_sample(w, counts):
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

    Rhist = shift_R(w, max_count, True)  # [T + 1, ]
    U = torch.rand_like(w[0]) * Rhist[-1].gather(0, counts.unsqueeze(0))[0]
    # N.B. counts is now going to double for n - r in Chen '97
    for k in range(1, T + 1):
        R = Rhist[-k - 1].gather(0, counts.unsqueeze(0))[0]
        match = U >= R
        b.insert(0, match)
        U = torch.where(match, (U - R) / w[T - k], U)
        counts = counts - match.long()

    del Rhist, U
    return torch.stack(b).float()
