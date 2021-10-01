"""Utility functions not otherwise specified"""

from typing import Tuple
import torch
import math


@torch.jit.script
@torch.no_grad()
def enumerate_bernoulli_support(
    tmax: int, lmax: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Enumerate all configurations of a specific number of Bernoulli highs

    Parameters
    ----------
    tmax : int
        The number of binary values per sequence
    lmax : torch.Tensor
        A long tensor of shape ``(nmax,)`` where ``lmax[n]`` is the number of highs
        in each configuration for the ``n``-th batch element

    Returns
    -------
    support, lens : torch.Tensor, torch.Tensor
        `support` is a tensor of shape ``(smax, tmax)`` where ``support[s]`` is the
        ``s``-th configuration and `lens` is a long tensor of shape ``(nmax,)``
        where ``lens[n]`` provides the number of sequential configurations belonging
        to batch element ``n``, i.e. all the configurations in
        ``support[lens[n - 1]: lens[n]]``.

        .. math ::

            s_n = \mathrm{Binom}(tmax, lmax) = \frac{tmax!}{(tmax - lmax)!lmax!} \\
            smax = sum_n s_n

    Warnings
    --------
    This function is only feasible for small values of `tmax` and `lmax` and is intended
    primarily for testing.
    """
    assert lmax.dim() == 1
    assert (lmax <= tmax).all() and (lmax >= 0).all()
    support = torch.zeros((tmax, int(2 ** tmax)), dtype=torch.bool, device=lmax.device)
    for t in range(tmax):
        support.view(tmax, int(2 ** t), 2, -1)[t, :, 0] = True
    accept_mask = support.sum(0, keepdim=True) == lmax.unsqueeze(1)  # (nmax, 2^tmax)
    lens = accept_mask.sum(1)
    support = (
        support.t().unsqueeze(0).masked_select(accept_mask.unsqueeze(2)).view(-1, tmax)
    )
    return support, lens


@torch.jit.script
@torch.no_grad()
def enumerate_gibbs_partition(b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert b.dim() == 2
    dtype = b.dtype
    b = b.t().to(torch.bool)
    nmax, tmax = b.size()
    lmax = b.sum(1)
    lmax_max = int(lmax.max())
    if lmax_max == 0:
        return b, torch.ones_like(lmax, dtype=torch.long)

    candidates = torch.empty(
        lmax_max + 1, nmax, tmax, tmax, dtype=torch.bool, device=b.device
    )
    valid = torch.empty(lmax_max + 1, nmax, tmax, dtype=torch.bool, device=b.device)

    # the first candidates are the originals themselves. This way we can exclude
    # replacing the event locations when searching for new candidates
    candidates[0, :, 0] = b
    valid[0] = False
    valid[0, :, 0] = True

    # e.g. b        = [1, 0, 0, 1, 1, 0, 0, 1]
    #      c        = [5, 1, 1, 6, 7, 3, 3, 8]
    #      cm1      = [0, 1, 1, 0, 0, 0, 0, 0]
    #      km1      = [0, 0, 0, 1, 1, 1, 1, 1]
    #      cm2      = [0, 1, 1, 0, 0, 0, 0, 0]
    #      km2      = [1, 0, 0, 0, 1, 1, 1, 1]
    #      cm3      = [0, 0, 0, 0, 0, 1, 1, 0]
    #      km3      = [1, 1, 1, 1, 0, 0, 0, 1]
    #      cm4      = [0, 0, 0, 0, 0, 1, 1, 0]
    #      km4      = [1, 1, 1, 1, 1, 0, 0, 0]
    c = b.cumsum(1) + b * lmax_max
    eye = torch.eye(tmax, device=b.device, dtype=torch.bool)
    for ell in range(1, lmax_max + 1):
        valid[ell] = (ell <= lmax).unsqueeze(1).expand(nmax, tmax)
        chng_msk = (c >= ell - 1) & (c <= ell)  # (nmax, tmax)
        keep_msk = (c < ell - 1) | (c > ell) & (c != ell + lmax_max)
        candidates[ell] = (chng_msk.unsqueeze(2) & eye) | (keep_msk & b).unsqueeze(1)
        a = candidates[ell].sum(2) == lmax.unsqueeze(1)
        valid[ell] = valid[ell] & a  # torchscript interpreter bug doesn't allow &=

    valid, candidates = valid.transpose(0, 1), candidates.transpose(0, 1)
    lens = valid.flatten(1).sum(1)
    chosen = candidates[valid].view(-1, tmax).t().to(dtype)
    return chosen, lens


# TESTS


def test_enumerate_bernoulli_support():
    def binom_coefficient(a: int, b: int) -> int:
        return math.factorial(a) // (math.factorial(a - b) * math.factorial(b))

    tmax = 10
    lmax = torch.arange(tmax + 1).repeat_interleave(2)
    support, lens = enumerate_bernoulli_support(tmax, lmax)
    assert lens.dim() == 1 and lens.numel() == 2 * (tmax + 1)
    assert support.size() == torch.Size([lens.sum().item(), tmax])
    for n in range(2 * (tmax + 1)):
        lmax_ = n // 2
        assert lens[n] == binom_coefficient(tmax, lmax_)
        support_n, support = support[: lens[n]], support[lens[n] :]
        assert (support_n.sum(1) == lmax_).all()


def test_enumerate_gibbs_partition():
    b = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).t()
    lens_exp = torch.tensor([9, 11, 1])
    chosen_exp = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # start 0
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # end 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # start 1
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # end 1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # start & end 2
        ]
    ).t()
    chosen_act, lens_act = enumerate_gibbs_partition(b)

    assert torch.equal(lens_exp, lens_act)
    assert torch.equal(chosen_exp, chosen_act)
