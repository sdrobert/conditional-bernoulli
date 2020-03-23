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


def shift_R(w, k_max, keep_hist=False):
    # in w = (T, *)
    # out R = (k_max + 1, *) if keep_hist is False, otherwise
    # R = (T + 1, k_max + 1, *)
    R0 = torch.ones_like(w[:1, ...])
    Rrest = torch.zeros((k_max,) + w[0].size(), device=w.device)
    w = w.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([R0, Rrest])]
    for T in range(w.shape[0]):
        Rrest = Rrest + w[T] * torch.cat([R0, Rrest[:-1]], 0)
        if keep_hist:
            hist.append(torch.cat([R0, Rrest]))
    if keep_hist:
        return torch.stack(hist, 0)
    else:
        return torch.cat([R0, Rrest])


def shift_log_R(logits, k_max, keep_hist=False):
    # in logits = (T, *)
    # out log_R = (k_max + 1, *)
    log_R0 = torch.zeros_like(logits[:1, ...])
    log_Rrest = torch.full(
        (k_max,) + logits[0].size(), -float('inf'), device=logits.device)
    logits = logits.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([log_R0, log_Rrest])]
    for T in range(logits.shape[0]):
        x = torch.cat([log_R0, log_Rrest[:-1]], 0)
        x = torch.where(torch.isfinite(x), x + logits[T], x)
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
    return shift_R(w, len(w)) / (1 + w).prod(0, keepdim=True)


def lprobs(logits):
    # in logits = (T, *)
    # out log_p = (T, *)
    return (
        shift_log_R(logits, len(logits)) +
        torch.nn.functional.logsigmoid(-logits).sum(0, keepdim=True)
    )
