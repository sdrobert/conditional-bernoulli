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
        return torch.empty(0, *orig_shape[1:], device=w.device, dtype=int)

    Rhist = R(w, max_count, True, True)  # (T + 1, max_count + 1, *)
    w = w.transpose(0, -1).flatten(end_dim=-2)  # (N, T)
    b = [torch.full_like(w[..., 0], -1, dtype=int)]  # [(N,)]
    Rhist = Rhist[:-1].flatten(2).transpose(0, 1).transpose(1, 2).flip(2)
    t = torch.arange(T, dtype=int, device=w.device)
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
        return torch.empty(0, *orig_shape[1:], device=logits.device, dtype=int)

    Rhist = lR(logits, max_count, True, True)  # (T + 1, max_count + 1, *)
    logits = logits.transpose(0, -1).flatten(end_dim=-2)  # (N, T)
    b = [torch.full_like(logits[..., 0], -1, dtype=int)]  # [(N,)]
    Rhist = Rhist[:-1].flatten(2).transpose(0, 1).transpose(1, 2).flip(2)
    t = torch.arange(T, dtype=int, device=logits.device)
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
