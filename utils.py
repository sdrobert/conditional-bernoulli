"""Utility functions not otherwise specified"""

from typing import Tuple
import torch
import math

__all__ = ["enumerate_bernoulli_support"]


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
