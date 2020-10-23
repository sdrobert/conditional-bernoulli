import torch


def R(w, k_max, keep_hist=False, reverse=False):
    # in w = (T, *)
    # out R = (k_max + 1, *) if keep_hist is False, otherwise
    # R = (T + 1, k_max + 1, *)
    R0 = torch.ones_like(w[:1, ...])
    Rrest = torch.zeros((k_max,) + w[0].size(), device=w.device)
    w = w.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([R0, Rrest])]
    T = w.shape[0]
    for t in range(T):
        if reverse:
            t = T - t - 1
        Rrest = Rrest + w[t] * torch.cat([R0, Rrest[:-1]], 0)
        if keep_hist:
            hist.append(torch.cat([R0, Rrest]))
    if keep_hist:
        return torch.stack(hist, 0)
    else:
        return torch.cat([R0, Rrest])


def lR(logits, k_max, keep_hist=False, reverse=False):
    # in logits = (T, *)
    # out log_R = (k_max + 1, *)
    log_R0 = torch.zeros_like(logits[:1, ...])
    log_Rrest = torch.full(
        (k_max,) + logits[0].size(), -float("inf"), device=logits.device
    )
    logits = logits.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([log_R0, log_Rrest])]
    T = logits.shape[0]
    for t in range(T):
        if reverse:
            t = T - t - 1
        x = torch.cat([log_R0, log_Rrest[:-1]], 0)
        # if x is infinite or logits is infinite, we don't want a gradient
        # FIXME(sdrobert): "where" is slow
        x = torch.where(
            torch.isfinite(x + logits[t]),
            x + logits[t],
            x.detach() + logits[t].detach(),
        )
        if k_max:
            log_Rrest = torch.logsumexp(torch.stack([log_Rrest, x], 0), 0)
        else:
            log_Rrest = x
        del x
        if keep_hist:
            hist.append(torch.cat([log_R0, log_Rrest]))
    if keep_hist:
        return torch.stack(hist, 0)
    else:
        return torch.cat([log_R0, log_Rrest])


def probs(w):
    # in w = (T, *)
    # out p = (T, *)
    return R(w, len(w)) / (1 + w).prod(0, keepdim=True)


def lprobs(logits):
    # in logits = (T, *)
    # out log_p = (T, *)
    return lR(logits, len(logits)) + torch.nn.functional.logsigmoid(-logits).sum(
        0, keepdim=True
    )
