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


def shift_log_R(theta, k_max):
    # in theta = (T, *)
    # out log_R = (k_max + 1, *)
    log_R0 = torch.zeros_like(theta[:1, ...])
    log_Rrest = torch.full(
        (k_max,) + theta[0].size(), -float('inf'), device=theta.device)
    theta = theta.unsqueeze(1)
    for T in range(theta.shape[0]):
        x = torch.cat([log_R0, log_Rrest[:-1]], 0)
        x = torch.where(torch.isfinite(x), x + theta[T], x)
        log_Rrest = torch.logsumexp(torch.stack([log_Rrest, x], 0), 0)
        del x
    return torch.cat([log_R0, log_Rrest])
