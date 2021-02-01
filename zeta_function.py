# Copyright 2021 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different implementations and benchmarking CLI for the zeta function"""

from typing import Callable, Optional, Union
import torch


def zeta(
    w: torch.Tensor,
    f: Union[Callable, torch.Tensor],
    L: Optional[int] = None,
    K: Optional[int] = None,
    full: bool = False,
) -> torch.Tensor:
    r"""The zeta function

    Choose some fixed-length history `K`, :math:`K' = \min(\ell, K)`,
    :math:`\{t_1, \ldots t_L\}` as event locations, then

    .. math::

        \zeta(L; w, f) = \sum_{\{t_1, \ldots, t_L : 1 \leq t_1 < \ldots < t_L \leq T\}}
            \prod_{\ell=1}^L w_{t_\ell} f_\ell(t_{\max(1,\ell - K + 1)}, \ldots, t_\ell)

    We consider it a generalization of the count function as we can retrieve the
    latter's values by setting :math:`K = 1`, :math:`f_\ell(t) \equiv 1`.

    To calculate :math:`\zeta`, we allow three ways to pass the values of
    :math:`f_\ell(\ldots)` viz. the argument `f`. We use :math:`n` in the superscript,
    :math:`f^n_\ell(u)`, to denote the (family of) functions for the ``n``-th batch
    element :math:`n \in [1, N]`

    1. If `f` is a function and `L` and `K` are defined, `f` accepts :math:`K' + 1`
       positional arguments as input and returns a tensor of shape ``N`` where
       :math:`\mathrm{f}(\ell, u_1, \ldots, u_{K'})[n - 1] = f^n_\ell(u)`. Note
       the distinction between the 1-indexed family :math:`f^n_\ell(u)` and the
       0-indexed batch index in `f`.
    2. If `f` is a function and only `L` is defined, `f` accepts one positional argument
       and returns a tensor of shape ``(N,) + (T,) * K'`` where
       :math:`\mathrm{f}(\ell)[n - 1, u_{K'} - 1, \ldots, u_1 - 1] = f^n_\ell(u)`.
       Note how :math:`u` is transformed into indices of `f`: the order of elements in
       :math:`u` are reversed, and  we subtract 1 from each :math:`u_k` to convert
       the 1-indexed arguments to :math:`f^n_\ell` into indices of `f`.
    3. `f` is a tensor of shape ``(L, N) + (T,) * K``, where :math:`\mathrm{f}[\ell - 1,
       n - 1, u_K - 1, \ldots, u_1 - 1] = f^n_\ell(u)`. Note subtracting 1 from
       :math:`\ell` to convert form 1-indexing to 0-indexing. Importantly, only changing
       the indices :math:`u_k` for :math:`k \leq K'` result in potentially different
       values; :math:`\forall k > K', u_k \neq u_k' : \mathrm{f}[\ell, n, \ldots, u_k] =
       \mathrm[\ell, n, \ldots, u'_k]`.

    Parameters
    ----------
    w : torch.Tensor
        A tensor of weights of shape ``(N, T)``
    f : function or torch.Tensor
    L : int or :obj:`None`, optional
    K : int or :obj:`None`, optional
    full : bool, optional
        Whether to return intermediary values. See return value below for more
        information.

    Returns
    -------
    Z : torch.Tensor
        If `full` is :obj:`False`, `Z` is tensor of shape ``(L + 1, N)`` where
        :math:`\mathrm{Z}[\ell, n - 1] = \zeta(\ell; w[n - 1], f^n)`. If :obj:`True`,
        `Z` is of shape ``(L + 1, N, T + 1)`` where
        :math:`\mathrm{Z}[\ell, n - 1, t] = \zeta(\ell; w[n - 1, :t], f^n)`.

    Warnings
    --------
    When method 3 is employed and :math:`K > 1`, it is important that, when
    :math:`f_\ell` does not have full context ():math:`K > K'`), the duplicate indices
    of `f` have the exact same value. Further, if gradients are desired, those values
    should all be attached and calculated in the same way (i.e. they will yield the same
    gradients). Method 3 effectively computes each duplicate as if it were unique, then
    normalizes :math:`\zeta'` according to the number of duplicate contexts.

    For example, suppose we've set :math:`K = 2` and we compute :math:`\mathrm{f_0}[n,
    u_1 - 1] = f^n_1(u_1)` of shape ``(N, T)`` separately from the rest of `f`,
    ``f_gt_0`` of shape ``(L - 1, N, T, T)``, that had full context. The below would
    yield an appropriate `f`:

    >>> f = torch.cat([
    ...     f_0.reshape(1, N, 1, T).expand(1, N, T, T),  # recall u_2 comes first
    ...     f_gt_0
    ... ], 0)  # (L, N, T, T)
    """
    if w.dim() != 2:
        raise RuntimeError("Expected w to be two dimensional")
    if callable(f):
        if L is None:
            raise RuntimeError("If f is a function, L must be specified")
        elif K is None:
            return _zeta_func_vectorized(w, f, L, full)
        elif K == 1:
            Z = _zeta_func_loop_k1(w.transpose(0, 1).contiguous(), f, L, full)
            return Z.transpose(1, 2) if full else Z
        else:
            return _zeta_func_loop(w, f, L, K, full)
    if f.dim() < 3:
        raise RuntimeError("If f is a tensor, f must be at least 3 dimensional")
    if L is not None and L != f.shape[0]:
        raise RuntimeError(
            "Expected first dimension of f ({}) to match L ({})".format(f.shape[0], L)
        )
    if K is not None and K + 2 != f.dim():
        raise RuntimeError(
            "Expected f to have K + 2 ({}) dimensions, got {}".format(K + 2, f.dim())
        )
    return _zeta_tensor(w, f, full)


def _zeta_func_loop_k1(
    w: torch.Tensor, f: Callable[[int, int], torch.Tensor], L: int, full: bool
) -> torch.Tensor:
    # When K=1, zeta has the recurrence
    # zeta(ell; w_{<= t}, f) = zeta(ell; w_{<t}, f) +
    #                               w_t f(ell, t) zeta(ell - 1; w_{<t}, f)
    # zeta(0; w_{<= t}, f) = 1
    T, N = w.shape
    ones = torch.ones((N,), device=w.device, dtype=w.dtype)
    zeros = torch.zeros((N,), device=w.device, dtype=w.dtype)
    Z_ell = [ones] * (T + 1)
    Z = [ones.unsqueeze(0).expand(T + 1, N)] if full else [ones]
    for ell in range(1, L + 1):
        Z_ellm1 = Z_ell
        Z_ell = [zeros] * ell
        for t in range(ell, T + 1):
            Z_ell.append(Z_ell[-1] + w[t - 1] * f(ell, t) * Z_ellm1[t - 1])
        Z.append(torch.stack(Z_ell, 0) if full else Z_ell[-1])
    return torch.stack(Z, 0)


def _zeta_func_loop(
    w: torch.Tensor, f: Callable, L: int, K: int, full: bool
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def _zeta_func_vectorized(
    w: torch.Tensor, f: Callable, L: int, full: bool
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def _zeta_tensor(w: torch.Tensor, f: torch.Tensor, full: bool) -> torch.Tensor:
    raise NotImplementedError("TODO")
