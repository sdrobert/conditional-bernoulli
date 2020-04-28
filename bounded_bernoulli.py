import torch
from poisson_binomial import R, lR


# this is more of a sanity check than anything. It should generate samples
# from the CB, but using the BB step function
def sample(w, counts):
    # in w = (T, *), counts = (*) or int
    # out b = (counts.max(), *)

    w = w.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=w.device)
    counts = counts.expand_as(w[0]).detach().flatten()  # (N,)

    orig_shape = w.shape
    T = orig_shape[0]
    max_count = counts.max().item()
    assert 0 <= max_count <= T
    if not max_count:
        return torch.empty(
            0, *orig_shape[1:], device=w.device, dtype=torch.long)

    Rhist = R(w, max_count, True, True)  # (T + 1, max_count + 1, *)
    w = w.transpose(0, -1).flatten(end_dim=-2)  # (N, T)
    b = [torch.full_like(w[..., 0], -1, dtype=torch.long)]  # [(N,)]
    Rhist = Rhist[:-1].flatten(2).transpose(0, 1).transpose(1, 2).flip(2)
    t = torch.arange(T, dtype=torch.long, device=w.device)
    # Rhist[c, :, t] = R(c, A^c_{t + 1})

    # P(t = t_l|L - l, t_{l-1}) = w_t R(L - l, A^c_t)
    #                             / R(L - l + 1, A^c_t_{l-1})
    #                           = w[:, t - 1] Rhist[L - l, :, t - 1] /
    #                             Rhist[L - l + 1, :, b[-1]]
    #                           = p[:, t - 1]
    for l in range(1, max_count + 1):
        Lml = counts - l
        R_Lml = Rhist.gather(
            0, Lml.clamp(0).unsqueeze(0).unsqueeze(-1).expand_as(Rhist[1:]))[0]
        t_lm1 = b[-1]

        p = (
            (w * R_Lml).masked_fill((t_lm1.unsqueeze(-1) >= t), 0.)
            .masked_fill((Lml < 0).unsqueeze(-1), 1.)
        )

        dist = torch.distributions.Categorical(probs=p)  # normalizes
        b.append(dist.sample().masked_fill(Lml < 0, -1))

    b = torch.stack(b[1:], 0).view(max_count, *orig_shape[1:])
    return b


def lsample(logits, counts):
    # in logits = (T, *), counts = (*) or int
    # out b = (counts.max(), *)

    logits = logits.detach()
    if not torch.is_tensor(counts):
        counts = torch.tensor(counts, device=logits.device)
    counts = counts.expand_as(logits[0]).detach().flatten()  # (N,)

    orig_shape = logits.shape
    T = orig_shape[0]
    max_count = counts.max().item()
    assert 0 <= max_count <= T
    if not max_count:
        return torch.empty(
            0, *orig_shape[1:], device=logits.device, dtype=torch.long)

    Rhist = lR(logits, max_count, True, True)  # (T + 1, max_count + 1, *)
    logits = logits.transpose(0, -1).flatten(end_dim=-2)  # (N, T)
    b = [torch.full_like(logits[..., 0], -1, dtype=torch.long)]  # [(N,)]
    Rhist = Rhist[:-1].flatten(2).transpose(0, 1).transpose(1, 2).flip(2)
    t = torch.arange(T, dtype=torch.long, device=logits.device)
    # Rhist[c, :, t] = R(c, A^c_{t + 1})

    # P(t = t_l|L - l, t_{l-1}) = w_t R(L - l, A^c_t)
    #                             / R(L - l + 1, A^c_t_{l-1})
    #                           = logits[:, t - 1] Rhist[L - l, :, t - 1] /
    #                             Rhist[L - l + 1, :, b[-1]]
    #                           = p[:, t - 1]
    for l in range(1, max_count + 1):
        Lml = counts - l
        R_Lml = Rhist.gather(
            0, Lml.clamp(0).unsqueeze(0).unsqueeze(-1).expand_as(Rhist[1:]))[0]
        t_lm1 = b[-1]

        lp = (
            (logits + R_Lml)
            .masked_fill((t_lm1.unsqueeze(-1) >= t), -float('inf'))
            .masked_fill((Lml < 0).unsqueeze(-1), 0.)
        )

        dist = torch.distributions.Categorical(logits=lp)  # normalizes
        b.append(dist.sample().masked_fill(Lml < 0, -1))

    b = torch.stack(b[1:], 0).view(max_count, *orig_shape[1:])
    return b


def probs(w, b):
    # in w = (T, *), b = (v, *)
    # out = (v, *)

    orig_shape = w.shape
    T = orig_shape[0]
    counts = (b >= 0).long().sum(0)
    max_count = counts.max().item()
    assert 0 <= max_count <= T
    if not max_count:
        return torch.zeros_like(b, dtype=w.dtype)

    Rhist = R(w, max_count, True, True)  # (T + 1, max_count + 1, *)

    # this function is mostly index selection. To avoid two gathers on Rhist,
    # we flatten arrays and perform a single take() per index selection.
    Rhist = Rhist.flatten()  # ((T + 1) * (max_count + 1) * N,)
    N = len(Rhist) // ((T + 1) * (max_count + 1))
    v = b.shape[0]
    counts = counts.flatten()  # (N,)
    w = w.flatten()  # (T * N,)
    b = b.flatten(1)  # (v, N)
    t_lm1 = torch.full_like(b[0], -1)  # (N,)
    N_range = torch.arange(N, dtype=torch.long, device=w.device)
    L_offs = N
    T_offs = (max_count + 1) * L_offs
    dummy_ps = torch.ones(N, device=w.device, dtype=w.dtype)

    p = []
    for l in range(1, max_count + 1):
        Lm1 = counts - l  # (N,)
        t_l = b[l - 1]  # (N,)
        R_t_l = Rhist.take(
            ((T - t_l - 1) * T_offs + Lm1 * L_offs + N_range).clamp(0))
        R_t_lm1 = Rhist.take(
            ((T - t_lm1 - 1) * T_offs + (Lm1 + 1) * L_offs + N_range).clamp(0))
        w_t_l = w.take((t_l * N + N_range).clamp(0))
        p.append(torch.where(Lm1 < 0, dummy_ps, w_t_l * R_t_l / R_t_lm1))
        t_lm1 = t_l

    p += [dummy_ps] * (v - max_count)
    return torch.stack(p, 0).view(v, *orig_shape[1:])


def lprobs(logits, b):
    # in logits = (T, *), b = (v, *)
    # out = (v, *)

    orig_shape = logits.shape
    T = orig_shape[0]
    counts = (b >= 0).long().sum(0)
    max_count = counts.max().item()
    assert 0 <= max_count <= T
    if not max_count:
        return torch.zeros_like(b, dtype=logits.dtype)

    Rhist = lR(logits, max_count, True, True)

    # this function is mostly index selection. To avoid two gathers on Rhist,
    # we flatten arrays and perform a single take() per index selection.
    Rhist = Rhist.flatten()  # ((T + 1) * (max_count + 1) * N,)
    N = len(Rhist) // ((T + 1) * (max_count + 1))
    v = b.shape[0]
    counts = counts.flatten()  # (N,)
    logits = logits.flatten()  # (T * N,)
    b = b.flatten(1)  # (v, N)
    t_lm1 = torch.full_like(b[0], -1)  # (N,)
    N_range = torch.arange(N, dtype=torch.long, device=logits.device)
    L_offs = N
    T_offs = (max_count + 1) * L_offs
    dummy_ps = torch.zeros(N, device=logits.device, dtype=logits.dtype)

    lp = []
    for l in range(1, max_count + 1):
        Lm1 = counts - l  # (N,)
        t_l = b[l - 1]  # (N,)
        R_t_l = Rhist.take(
            ((T - t_l - 1) * T_offs + Lm1 * L_offs + N_range).clamp(0))
        R_t_lm1 = Rhist.take(
            ((T - t_lm1 - 1) * T_offs + (Lm1 + 1) * L_offs + N_range).clamp(0))
        logits_t_l = logits.take((t_l * N + N_range).clamp(0))
        lp.append(torch.where(Lm1 < 0, dummy_ps, logits_t_l + R_t_l - R_t_lm1))
        t_lm1 = t_l

    lp += [dummy_ps] * (v - max_count)
    return torch.stack(lp, 0).view(v, *orig_shape[1:])
