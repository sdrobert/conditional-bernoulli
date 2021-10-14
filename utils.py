# Copyright 2021 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    r"""Enumerate the possible ell-th event locations given the other ones over all ell

    The Gibbs-style conditional density of a single event location given the rest for
    a fixed number of events is

    .. math::
        \pi(\tau_\ell = t|\tau_{-\ell}) = \frac{P(\tau_{-\ell}(t))G(\tau_{-\ell}(t))}
                                      {\sum_{t'}P(\tau_{-\ell}(t'))G(\tau_{-\ell}(t'))}
    
    Where :math:`\tau_{-\ell}` is some configuration of event locations excluding the
    :math:`\ell`-th and :math:`\tau_{-\ell}(t)` is the :math:`\tau` such that
    :math:`\tau_\ell = t` and the rest are :math:`\tau_{-\ell}`.

    Letting :math:`\tau` characterize the ``n``-th batched element in `b`, ``b[:, n]``,
    this function determines all possible :math:`\tau'` (and thus :math:`b'`, `bp`),
    which would contribute to the numerator and/or denominator of :math:`\pi(\tau_\ell =
    t|\tau_{-\ell})` for any given :math:`\ell`.

    Parameters
    ----------
    b : torch.Tensor
        Of shape ``(tmax, nmax)``
    
    Returns
    -------
    bp, lens : torch.Tensor, torch.Tensor
    
        `lens` is a long tensor of shape ``(nmax, lmax_max)``, where
        ``lmax_max = b.sum(0).max()`` is the maximum number of events in any batched
        element in `b`. `bp` of shape ``(tmax, nmax*)``, where ``nmax*`` is the total
        number of :math:`b'` contributing to any :math:`\pi(\tau_\ell = t|\tau_{-\ell})`
        in any batch element, *except* those where :math:`b' = b`. :math:`b` has a
        nonzero contribution to any :math:`\pi(\tau_\ell = t|\tau_{-\ell})` (assuming
        :math:`|\tau| > 0`), but is excluded to avoid redundancy. ``lens[n, ell - 1]``
        is the number of :math:`b'` contributiong to
        :math:`\pi(\tau_\ell = t|\tau_{-\ell})` associated with batch element
        ``b[:, n]``. The :math:`b'` themselves are stored in `bp`, contiguous first in
        increasing :math:`\ell` and then in ``n``. For example, the configurations
        for ``n = 1`` and ``ell - 1 = 1`` are stored in
        
            bp[:, lens[0].sum() + lens[1, 0]:lens[0].sum() + lens[1, :1].sum()]
    """
    assert b.dim() == 2
    device, dtype = b.device, b.dtype
    b = b.t().to(torch.bool)
    nmax, tmax = b.size()
    lmax = b.sum(1)
    lmax_max = int(lmax.max())
    if not lmax_max:
        return (
            torch.empty(tmax, 0, dtype=dtype, device=device),
            torch.empty(nmax, 0, dtype=torch.long, device=device),
        )

    candidates = torch.empty(
        lmax_max, nmax, tmax, tmax, dtype=torch.bool, device=b.device
    )
    valid = torch.empty(lmax_max, nmax, tmax, dtype=torch.bool, device=b.device)

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
    for ell in range(lmax_max):
        valid[ell] = (ell < lmax).unsqueeze(1).expand(nmax, tmax)
        chng_msk = (c >= ell) & (c <= ell + 1)  # (nmax, tmax)
        keep_msk = (c < ell) | (c > ell + 1) & (c != ell + lmax_max + 1)
        candidates[ell] = (chng_msk.unsqueeze(2) & eye) | (keep_msk & b).unsqueeze(1)
        a = candidates[ell].sum(2) == lmax.unsqueeze(1)
        valid[ell] = valid[ell] & a  # torchscript interpreter bug doesn't allow &=

    valid, candidates = valid.transpose(0, 1), candidates.transpose(0, 1)
    lens = valid.sum(2)  # (nmax, lmax)
    bp = candidates[valid].view(-1, tmax).t().to(dtype)
    return bp, lens


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
    lens_exp = torch.tensor([[2, 2, 2, 2], [5, 5, 0, 0], [0, 0, 0, 0]])
    # N.B. bp excludes b itself.
    bp_exp = torch.tensor(
        [
            # n=0, ell=0
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            # ell=1
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            # ell=2
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            # ell=3
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            # n=1, ell=0
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            # n=1, ell=1
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    ).t()
    bp_act, lens_act = enumerate_gibbs_partition(b)

    assert torch.equal(lens_exp, lens_act)
    assert torch.equal(bp_exp, bp_act)
