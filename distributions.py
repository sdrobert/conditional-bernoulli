import torch


def generalized_binomial(logits, k_max, intermediate=None):
    r'''Calculate the generalized binomial coefficient for log-odds

    For a set of odds :math:`w_t \in [0, 1]` and :math:`t \in [1, T]`, the
    generalized binomial coefficient for some number of choices :math:`k` is
    defined as

    .. math::

        C(k, w) = \sum_{\{S \subseteq T: |S| = k\}} \prod_{t \in S} w_t

    This function computes :math:`log C(k, w)` using
    :math:`\text{logits}_t = \log w_t`. `logits` is a float vector of shape
    ``(T,)`` or a matrix of shape ``(T, N)``, where ``T`` is the choice
    dimension and ``N`` is the batch dimension. `k_max` is an integer of the
    maximum `k` value to compute.

    If `intermediate` is :obj:`None` (the default), this function returns `lC`,
    which is either a vector of shape ``(k_max + 1,)`` or a matrix of shape
    ``(k_max + 1, N)``, depending on whether `logits` was batched.
    :math:`lC[k] = \log C(k, w)`.

    If `intermediate` is either ``"forward"`` or ``"reverse"``, `Cl` is a
    tensor of shape ``(T + 1, k.max() + 1[, N])`` that stores all intermediate
    values used in the computation. If ``"forward"``,
    :math:`lC[t, k'] = \log C(k', w_{\leq t})`. If ``"reverse"``, the mapping
    :math:`w_t \mapsto \tilde{w}_{T - t + 1}` is first applied before the
    forward procedure.

    Parameters
    ----------
    logits : torch.tensor
    k_max : int
    intermediate : {None, "forward", "reverse"}

    Returns
    -------
    lC : torch.tensor
    '''
    # Implementation note: this is the "direct" method from
    # Chen and Liu, 1997. "Statistical applications of the Poisson-Binomial and
    # Conditional Bernoulli distributions"

    if logits.dim() == 1:
        flatten = True
        logits = logits.unsqueeze(-1)
    else:
        flatten = False
    if logits.dim() != 2:
        raise RuntimeError('logits must be one or two dimensional')
    T, N = logits.shape

    if k_max < 0:
        raise RuntimeError('k_max must be >= 0 (note: log C(<0, *) = -inf)')

    if intermediate not in {None, "forward", "reverse"}:
        raise RuntimeError(
            "intermediate must be one of None, 'forward', or 'reverse'")

    if not k_max or not T:
        Cl = torch.full(
            (1, N), 0 if not k_max else -float('inf'),
            device=logits.device, dtype=logits.dtype)
        if intermediate:
            Cl = Cl.expand(T + 1, 1, N)
        if flatten:
            Cl = Cl.squeeze(1)
        return Cl

    lC_first = torch.zeros(1, N, device=logits.device, dtype=logits.dtype)
    lC_rest = torch.full(
        (k_max, N), -float('inf'), device=logits.device, dtype=logits.dtype)
    if intermediate:
        hist = [torch.cat([lC_first, lC_rest])]

    logits = logits.unsqueeze(1)
    for t in range(T):
        if intermediate == 'reverse':
            t = T - t - 1

        x = torch.cat([lC_first, lC_rest[:-1]], 0)
        # if x is infinite or logits is infinite, we don't want a gradient
        x = torch.where(
            torch.isfinite(x + logits[t]),
            x + logits[t],
            x.detach() + logits[t].detach()
        )
        lC_rest = torch.logsumexp(torch.stack([lC_rest, x]), 0)
        del x

        if intermediate:
            hist.append(torch.cat([lC_first, lC_rest]))

    if intermediate:
        Cl = torch.stack(hist, 0)
    else:
        Cl = torch.cat([lC_first, lC_rest])
    if flatten:
        Cl = Cl.squeeze(-1)
    return Cl
