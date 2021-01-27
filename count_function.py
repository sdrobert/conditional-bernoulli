# Copyright 2020 Sean Robertson
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

"""Different implementations and benchmarking CLI for the count function"""

import torch
import contextlib
import math
import argparse
import numpy as np
import fractions


def count(
    w, L, method="cumsum", include_hist=False, direction="forward", batch_first=True
):
    r"""The count function

    The count function is defined as

    .. math::

        C(L; w) = \sum_{\{A \subseteq [1, T] : |A| = L\}} \prod_{a \in A} w_a

    It can be considered a generalization of the binomial coefficient to unequal
    weights.

    Parameters
    ----------
    w : torch.Tensor
        A tensor of shape ``(N, T)`` if `batch_first` is :obj:`True`, else ``(T, N)``.
    L : int
        A non-negative integer representing the maximum product to count up to.
    method : str, optional
        The method/algorithm used in computing the count. See below for a list of all
        available algorithms and their descriptions.
    include_hist : bool, optional
        If :obj:`True`, will return intermediate counts from subsets of `w`.
    direction : {'forward', 'backward', 'both'}, optional
        Whether to compute counts using subsets of prefixes of weights ``w_1 ... w_t``
        (when ``'forward'``), suffixes of weights ``w_t ... w_T`` (when ``'backward'``),
        or do ``'both'``. ``'both'`` will affect the size of the returned tensor, but
        otherwise this has no impact on the return value when `include_hist` is
        :obj:`False`.
    batch_first : bool, optional
        Whether the 0-index of `w` is the batch index or not

    Returns
    -------
    C : torch.Tensor
        By default, `C` is a tensor of shape ``(N, L + 1)`` where ``C[n, ell] = C(ell;
        w[n])``. If `include_hist` is :obj:`True`, `C` is of shape ``(N, L
        + 1, T + 1)`` where ``C[n, ell, t] = C(ell; {w[n, 0] ... w[n, t - 1]})``. If
        `include_hist` is :obj:`True` and `direction` is ``'backward'``, ``C[n, ell, t]
        = C(ell; {w[n, T - t] .. w[n, T - 1]})``. If :obj:`True` and ``'both'``, `C` is
        of shape ``(2, N, L + 1, T + 1)``, where ``C[0]`` stores the forward counts and
        ``C[1]`` the backward count. If `batch_first` is :obj:`False`, `C` will have
        shape ``(L + 1, N)`` by default, ``(T + 1, L + 1, N)`` if including history, and
        ``(T + 1, L + 1, 2, N)`` if including history in both directions.

    Notes
    -----
    The following is a list of available methods

    -  'chen94': The recursive method of theorem 3 of Chen, X. et al (1994)
       "Weighted finite population sampling to maximize entropy."
    -  'howard97': The recursive method outlined in Howard, S. (1972) "Discussion
       on professor Cox's paper." Iterates over the ``T`` dimension in `C` and
       parallelizes over ``L``.
    -  'full_matrix': Based off the recursion of Howard, but iterates over the ``L``
       dimension in `C`, calculating the ``T`` dimension by matrix multiplication.
    -  'block_matrix': Same as 'full_matrix', except performs block matrix
       multiplication to use only half the memory.
    -  'cumsum': Similar to 'full_matrix', but takes advantage of :func:`torch.cumsum`
       to avoid the matrix multiplication.
    """
    if w.dim() != 2:
        raise RuntimeError("Expected w to be two dimensional")
    if L < 0:
        raise RuntimeError("L must be non-negative")
    if method not in _METHODS:
        raise RuntimeError("Invalid method {}".format(method))
    func, needs_batch_first = _METHODS[method]
    if needs_batch_first != batch_first:
        w = w.transpose(0, 1)
    if direction == "backward":
        w = w.flip(1 if needs_batch_first else 0)
    elif direction == "both":
        if needs_batch_first:
            w = torch.cat([w, w.flip(1)], 0)
        else:
            w = torch.cat([w, w.flip(0)], 1)
    elif direction != "forward":
        raise RuntimeError("Invalid direction {}".format(direction))
    w = w.contiguous()
    C = func(w, L, include_hist)
    if needs_batch_first != batch_first:
        C = C.transpose(0, -1)
    if direction == "both":
        C = C.contiguous()
        if batch_first:
            N2 = C.shape[0]
            C = C.view(*((2, N2 // 2) + C.shape[1:]))
        else:
            N2 = C.shape[-1]
            C = C.view(*(C.shape[:-1] + (2, N2 // 2)))
    return C


def _chen94(w, L, include_hist):
    # Chen's 1994 method from theorem 3. Note that here S[:, i] == T(i, w)
    # from the paper

    # pre-compute S values
    S = [(w ** ell).sum(1) for ell in range(1, L + 1)]

    # do the actual recursion in the helper
    C = _chen94_helper(L, S)

    if include_hist:
        # Chen's method produces C(L, w) for *only* the full set w. If we want more,
        # we have to re-calculate the recursion. However, as per Chen, we can avoid some
        # re-computation if we subtract terms from S
        C = [C]
        for t in range(w.shape[1] - 1, -1, -1):
            w_t = w[:, t]
            # remove the last weight, update S, and perform the recursion
            S = [S[ell - 1] - w_t ** ell for ell in range(1, L + 1)]
            C.insert(0, _chen94_helper(L, S))
        C = torch.stack(C, -1)

    return C


def _chen94_helper(L, S):
    N = S[0].shape[0]
    C = [torch.ones((N,), device=S[0].device, dtype=S[0].dtype)]
    zeros = torch.zeros((N,), device=S[0].device, dtype=S[0].dtype)
    for ell in range(1, L + 1):
        C_ell = zeros
        for i in range(1, ell + 1):
            if i % 2:
                C_ell = C_ell + S[i - 1] * C[ell - i]
            else:
                C_ell = C_ell - S[i - 1] * C[ell - i]
        C_ell = C_ell / ell
        C.append(C_ell)
    return torch.stack(C, -1)  # (N, L + 1)


def _howard72(w, L, include_hist):
    # recursion from Howard's 72 paper.
    # C(L, w) = C(L, w \ {w_t}) + w_t C(L - 1, w \ {w_t})
    # iteratively calculates C(\cdot, w_{\leq t}) using C(\cdot, w_{<t})
    T, N = w.shape
    C_t_first = torch.ones((1, N), dtype=w.dtype, device=w.device)
    C_t_rest = torch.zeros((L, N), dtype=w.dtype, device=w.device)
    C_tm1 = torch.cat([C_t_first, C_t_rest], 0)  # C_0
    if include_hist:
        C = [C_tm1]
    for t in range(T):
        C_t_rest = C_t_rest + w[t] * C_tm1[:-1]
        C_tm1 = torch.cat([C_t_first, C_t_rest], 0)
        if include_hist:
            C.append(C_tm1)
    if include_hist:
        return torch.stack(C, 0)
    else:
        return C_tm1


def _full_matrix(w, L, include_hist):
    # uses same recusion as _howard72, but iteratively calculates
    # C(L, w) using C(L - 1, w). Uses a matrix-vector multiplication
    N, T = w.shape
    w = torch.cat(
        [w, torch.zeros((N, 1), device=w.device, dtype=w.dtype)], -1
    )  # (N, T + 1)
    w = w.view(N, 1, T + 1).expand(N, T + 1, T + 1).tril(-1)  # (N, T + 1, T + 1)
    C_ell = torch.ones((N, T + 1), device=w.device, dtype=w.dtype)
    if include_hist:
        C = [C_ell]
    else:
        C = [C_ell[:, -1]]
    for _ in range(L):
        C_ell = torch.bmm(w, C_ell.unsqueeze(-1)).squeeze(-1)
        if include_hist:
            C.append(C_ell)
        else:
            C.append(C_ell[:, -1])
    C = torch.stack(C, 1)
    return C


def _block_matrix(w, L, include_hist):
    # same as _full_matrix, but splits full matrix multiplication into 3 blocks:
    #
    #     d           d
    #     |           |
    #     v           v
    # 0 0 0 0 0     0 0 0 0
    # 1 0 0 0 0     1 0 0 0
    # 1 1 0 0 0     2 2 0 0
    # 2 2 3 0 0     2 2 3 0
    # 2 2 3 3 0
    #
    # the rows in the second block are the same, so we only have a row of it
    N, T = w.shape
    d = T // 2 + 1
    B2_row = w[:, :d]
    B1 = B2_row.view(N, 1, d).expand(N, d, d).tril(-1)  # (N, d, d)
    B3 = (
        torch.cat([w[:, d:], torch.zeros((N, 1), device=w.device, dtype=w.dtype)], -1)
        .view(N, 1, T + 1 - d)
        .expand(N, T + 1 - d, T + 1 - d)
        .tril(-1)
    )  # (N, T + 1 - d, T + 1 - d)
    C_ell_top = torch.ones((N, d), device=w.device, dtype=w.dtype)  # (N, d)
    C_ell_bot = torch.ones(
        (N, T + 1 - d), device=w.device, dtype=w.dtype
    )  # (N, T + 1 - d)
    if include_hist:
        C = [torch.cat([C_ell_top, C_ell_bot], 1)]  # [(N, T + 1)]
    else:
        C = [C_ell_bot[:, -1]]  # [(N,)]
    for _ in range(L):
        C_ell_bot = (B2_row * C_ell_top).sum(1, keepdim=True).expand(
            N, T + 1 - d
        ) + torch.bmm(B3, C_ell_bot.unsqueeze(-1)).squeeze(-1)
        C_ell_top = torch.bmm(B1, C_ell_top.unsqueeze(-1)).squeeze(-1)
        if include_hist:
            C.append(torch.cat([C_ell_top, C_ell_bot], 1))
        else:
            C.append(C_ell_bot[:, -1])
    C = torch.stack(C, 1)
    return C


def _cumsum(w, L, include_hist):
    # similar to _full_matrix and _block_matrix in recursion, but uses cumsum instead of
    # matrix multiplication
    N, T = w.shape
    C_ell = torch.ones((N, T + 1), device=w.device, dtype=w.dtype)
    zeros = torch.zeros((N, 1), device=w.device, dtype=w.dtype)
    if include_hist:
        C = [C_ell]
    else:
        C = [C_ell[:, -1]]
    for _ in range(L):
        C_ell = torch.cat([zeros, torch.cumsum(w * C_ell[:, :-1], 1)], -1)
        if include_hist:
            C.append(C_ell)
        else:
            C.append(C_ell[:, -1])
    C = torch.stack(C, 1)
    return C


_METHODS = {
    "chen94": (_chen94, True),
    "howard72": (_howard72, False),
    "full_matrix": (_full_matrix, True),
    "block_matrix": (_block_matrix, True),
    "cumsum": (_cumsum, True),
}


def log_count(
    lw,
    L,
    method="log_cumsum",
    include_hist=False,
    direction="forward",
    batch_first=True,
):
    """The log-valued count function

    Perform the count function computations of :func:`count_function`, but try to stay
    in log-space, i.e. ``count(w, ...) ~= log_count(w.log(), ...).exp()``. This function
    is more numerically stable than that.

    Parameters
    ----------
    lw : torch.Tensor
    L : int
    method : {'log_chen94', 'log_howard72', 'log_cumsum'}, optional
        Identical methods as their counterparts in :func:`count`, except they use
        log-domain operations. Fewer methods are available.
    include_hist : bool, optional
    direction : {'forward', 'backward', 'both'}, optional
    batch_first : bool, optional

    Returns
    -------
    lC : torch.Tensor
    """
    if lw.dim() != 2:
        raise RuntimeError("Expected lw to be two dimensional")
    if L < 0:
        raise RuntimeError("L must be non-negative")
    if method not in _LOG_METHODS:
        raise RuntimeError("Invalid method {}".format(method))
    func, needs_batch_first = _LOG_METHODS[method]
    if needs_batch_first != batch_first:
        lw = lw.transpose(0, 1)
    if direction == "backward":
        lw = lw.flip(1 if needs_batch_first else 0)
    elif direction == "both":
        if needs_batch_first:
            lw = torch.cat([lw, lw.flip(1)], 0)
        else:
            lw = torch.cat([lw, lw.flip(0)], 1)
    elif direction != "forward":
        raise RuntimeError("Invalid direction {}".format(direction))
    lw = lw.contiguous()
    lC = func(lw, L, include_hist)
    if needs_batch_first != batch_first:
        lC = lC.transpose(0, -1)
    if direction == "both":
        lC = lC.contiguous()
        if batch_first:
            N2 = lC.shape[0]
            lC = lC.view(*((2, N2 // 2) + lC.shape[1:]))
        else:
            N2 = lC.shape[-1]
            lC = lC.view(*(lC.shape[:-1] + (2, N2 // 2)))
    return lC


# for log count functions, we want -inf to behave the same way as if it were zero in
# the regular count function. Thus we replace any occurrence with a log value that,
# when exponentiated, is nonzero, but only just. The division by two here ensures that
# we can add two EPS_INF values without the result being zero.
EPS_INF = math.log(torch.finfo(torch.float32).tiny) / 2


try:
    _logaddexp = torch.logaddexp
except AttributeError:

    def _logaddexp(a, b):
        max_ = torch.max(a, b)
        diff = -(a - b).abs()
        return max_ + diff.exp().log1p()


try:
    _logcumsumexp = torch.logcumsumexp
except AttributeError:

    def _logcumsumexp(x, dim):
        max_ = x.max(dim, keepdim=True)[0]
        x = -(x - max_).abs()
        return max_ + x.exp().cumsum(dim).log()


def _logsubexp(a, b):
    x = (-(b - a).expm1()).log() + a
    # x = (-(b - a).exp()).log1p() + a
    # x = (a.exp() - b.exp()).log()
    return x.masked_fill_(b >= a, EPS_INF)


def _log_chen94(lw, L, include_hist):

    lS = [(lw * ell).logsumexp(1) for ell in range(1, L + 1)]

    lC = _log_chen94_helper(L, lS)

    if include_hist:
        lC = [lC]
        for t in range(lw.shape[1] - 1, -1, -1):
            lw_t = lw[:, t]
            # remove the last weight, update S, and perform the recursion
            lS = [_logsubexp(lS[ell - 1], lw_t * ell) for ell in range(1, L + 1)]
            lC.insert(0, _log_chen94_helper(L, lS))
        lC = torch.stack(lC, -1)

    return lC


def _log_chen94_helper(L, lS):
    N = lS[0].shape[0]
    lC = [torch.zeros((N,), device=lS[0].device, dtype=lS[0].dtype)]
    ninfs = torch.full((N,), EPS_INF, device=lS[0].device, dtype=lS[0].dtype)
    for ell in range(1, L + 1):
        lC_ell = ninfs
        for i in range(1, ell + 1):
            if i % 2:
                lC_ell = _logaddexp(lC_ell, lS[i - 1] + lC[ell - i])
            else:
                lC_ell = _logsubexp(lC_ell, lS[i - 1] * lC[ell - i])
        lC_ell = lC_ell - math.log(ell)
        lC.append(lC_ell)
    return torch.stack(lC, -1)  # (N, L + 1)


def _log_howard72(lw, L, include_hist):
    T, N = lw.shape
    lC_t_first = torch.zeros((1, N), dtype=lw.dtype, device=lw.device)
    lC_t_rest = torch.full((L, N), EPS_INF, dtype=lw.dtype, device=lw.device)
    lC_tm1 = torch.cat([lC_t_first, lC_t_rest], 0)  # lC_0
    if include_hist:
        lC = [lC_tm1]
    for t in range(T):
        lC_t_rest = _logaddexp(lC_t_rest, lw[t] + lC_tm1[:-1])
        lC_tm1 = torch.cat([lC_t_first, lC_t_rest], 0)
        if include_hist:
            lC.append(lC_tm1)
    if include_hist:
        return torch.stack(lC, 0)
    else:
        return lC_tm1


def _log_cumsum(lw, L, include_hist):
    N, T = lw.shape
    lC_ell = torch.zeros((N, T + 1), device=lw.device, dtype=lw.dtype)
    lw = lw.clamp(min=EPS_INF)
    ninfs = torch.full((N, 1), EPS_INF, device=lw.device, dtype=lw.dtype)
    if include_hist:
        lC = [lC_ell]
    else:
        lC = [lC_ell[:, -1]]
    for _ in range(L):
        lC_ell = torch.cat([ninfs, _logcumsumexp(lw + lC_ell[:, :-1], 1)], -1)
        if include_hist:
            lC.append(lC_ell)
        else:
            lC.append(lC_ell[:, -1])
    lC = torch.stack(lC, 1)
    return lC


_LOG_METHODS = {
    "log_chen94": (_log_chen94, True),
    "log_howard72": (_log_howard72, False),
    "log_cumsum": (_log_cumsum, True),
}


def main(args=None):
    """Benchmark different count approximations"""
    opts = _parse_args(args)

    if opts.command == "speed":
        return _speed(opts)
    elif opts.command == "acc":
        return _accuracy(opts)


def _parse_args(args):
    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument("method", choices=sorted(_METHODS) + sorted(_LOG_METHODS))

    parser.add_argument("--log2-repeat", type=int, default=10)
    parser.add_argument("--log2-trials", type=int, default=10)
    parser.add_argument("--log2-highs", type=int, default=4)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--double", action="store_true", default=False)

    subparsers = parser.add_subparsers(dest="command", required=True)

    speed_parser = subparsers.add_parser("speed")
    speed_parser.add_argument("--log2-burn-in", type=int, default=10)
    speed_parser.add_argument("--hist", action="store_true", default=False)
    speed_parser.add_argument("--log2-batch", type=int, default=0)

    acc_parser = subparsers.add_parser("acc")
    acc_parser.add_argument("--seed", type=int, default=None)
    acc_parser.add_argument("--log2-ratio-odds", type=int, default=0)
    acc_parser.add_argument("--log2-expectation", type=int, default=0)

    return parser.parse_args(args)


@contextlib.contextmanager
def _cuda_timer():
    class Timer(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __call__(self):
            return self.start.elapsed_time(end)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timer = Timer(start, end)

    start.record()
    try:
        yield timer
    finally:
        end.record()
        torch.cuda.synchronize()


@contextlib.contextmanager
def _cpu_timer():
    class Timer(object):
        def __init__(self):
            self.start = self.end = None

        def __call__(self):
            return self.end - self.start

    timer = Timer()

    import time

    start = time.time() * 1000.0
    try:
        yield timer
    finally:
        end = time.time() * 1000.0
        timer.start, timer.end = start, end


def _speed(opts):
    cum_times = []

    if opts.device.type == "cuda":
        timeit = _cuda_timer
    else:
        timeit = _cpu_timer

    dtype = torch.float64 if opts.double else torch.float32
    T = 2 ** opts.log2_trials
    burn_in = 2 ** opts.log2_burn_in
    N = 2 ** opts.log2_batch
    L = 2 ** opts.log2_highs
    repeat = 2 ** opts.log2_repeat

    w = torch.zeros(N, T, device=opts.device, dtype=dtype)
    if opts.method.startswith("log"):
        func, batch_first = _LOG_METHODS[opts.method]
    else:
        func, batch_first = METHODS[opts.method]
    if not batch_first:
        w = w.transpose(0, 1).contiguous()
    if opts.device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(burn_in):
        func(w, L, opts.hist)

    for repeat_no in range(repeat):

        with timeit() as timer:
            v = func(w, L, opts.hist)
        assert not torch.isnan(v).any() and not torch.isinf(v).any()

        cum_times.append(timer())

    cum_times = np.array(cum_times)

    print(
        f"log2-repeat={opts.log2_repeat},"
        f"log2-trials={opts.log2_trials},"
        f"log2-highs={opts.log2_highs},"
        f"device={opts.device.type},"
        f"method={opts.method},"
        f"double={opts.double},"
        f"log2-burn-in={opts.log2_burn_in},"
        f"hist={opts.hist},"
        f"log2-batch={opts.log2_batch},"
        f"mean-time(std)={cum_times.mean():.4f}({cum_times.std():.4f})"
    )


def _binom(T, L):
    return math.factorial(T) // (math.factorial(T - L) * math.factorial(L))


def _accuracy(opts):
    diffs = []
    dhigh_diffs = []
    dmid_diffs = []
    dlow_diffs = []
    w_highs = []
    w_mids = []
    w_lows = []

    dtype = torch.float64 if opts.double else torch.float32
    T = 2 ** opts.log2_trials
    L = 2 ** opts.log2_highs
    repeat = 2 ** opts.log2_repeat
    ratio_odds = 2 ** opts.log2_ratio_odds
    V = 2 ** opts.log2_expectation

    if opts.method.startswith("log_"):
        func, batch_first = _LOG_METHODS[opts.method]
    else:
        func, batch_first = METHODS[opts.method]

    # this method is similar to that of https://doi.org/10.1016/j.csda.2012.10.006 but
    # is essentially prop 1.c of 10.2307/2337119, a generalization of Vandermonde's
    # identity. the intuition behind the identity is: if you split your odds into two
    # mutually exclusive subsets, w.l.o.g. w_{<=t_left} and w_{> t_right}, then the
    # number of ways you can chooose L from them is the sum over ell_left of the number
    # of ways to choose ell_left from w_{<=t_left} and L - ell_left from w_{>t_right}.
    # You can then apply the proposition again to split the region of w_{>t_left} into
    # w_{(t_left,t_right]} and w_{>t_right} and sum the remaining ways to choose
    # ell_mid from L - ell_left
    #
    # If all weights in w_{<=t_left} are identically w_high, t_left-choose-ell_left of
    # that subset is simply binom(t_left, ell_left) * w_high ** ell_left. Setting the
    # w_{(t_left,t_right]} identically to w_mid gives binom(t_right - t_left, ell_mid) *
    # w_mid ** ell_mid. Finally, setting w_{>t_right} identically to w_low gives
    # binom(T - t_right, L - ell_mid - ell_left) * w_low ** (L - ell_mid - ell_left).
    #
    # we set w_mid = w_high / (2 ** opts.log2_ratio_odds) and w_low = w_mid / (2 **
    # opts.log2_ratio_odds). We determine w_high as the solution where the total
    # expectation is equal to 2 ** opts.log2_expectation

    M = torch.tensor([3 / 2 * (T + 1), 3 * (T + 1)])

    if opts.seed is not None:
        torch.manual_seed(opts.seed)
    for repeat_no in range(repeat):
        if opts.seed is not None:
            torch.manual_seed(opts.seed + repeat_no)

        # use rejection sampling to pick a t_left and t_right such that
        # t_left <= t_right and E_{t_left,t_right}[t_left] ~= T / 3 and
        # E_{t_left,t_right}[t_right] ~= 2T / 3
        while True:
            t_left, t_right = (torch.rand(2) * M).int().tolist()
            if t_right < T + 1 and t_left <= t_right:
                break
        # sizes of the partitions
        T_left = t_left
        T_mid = t_right - t_left
        T_right = T - t_right
        T_noleft = T - t_left

        # determine the expected value assuming w_low == 1
        expected = fractions.Fraction(0)
        dleft_expected = fractions.Fraction(0)
        dmid_expected = fractions.Fraction(0)
        dright_expected = fractions.Fraction(0)
        # 0 <= ell_left <= L
        # T_left >= ell_left
        # T_noleft >= L - ell_left --> ell_left >= L - T_noleft
        for ell_left in range(max(0, L - T_noleft), min(L, T_left) + 1):
            L_noleft = L - ell_left
            contrib_left = _binom(T_left, ell_left)
            if T_left > 0 and ell_left > 0:
                dcontrib_left = _binom(T_left - 1, ell_left - 1)
            else:
                dcontrib_left = 0
            # 0 <= ell_mid <= L_noleft
            # T_mid >= ell_mid
            # T_right >= L_noleft - ell_mid --> ell_mid >= L_noleft - T_right
            for ell_mid in range(max(0, L_noleft - T_right), min(L_noleft, T_mid) + 1):
                ell_right = L_noleft - ell_mid
                contrib_mid = fractions.Fraction(
                    _binom(T_mid, ell_mid), ratio_odds ** ell_mid
                )
                contrib_right = fractions.Fraction(
                    _binom(T_right, ell_right), ratio_odds ** (2 * ell_right)
                )
                expected += contrib_left * contrib_mid * contrib_right
                # the derivative with respect to one of the weights merely fixes that
                # choice of weight and replaces its value with one. Effectively, we
                # choose one less term from a pool of size one less. If
                # t_left < t <= t_right, we choose one less from the middle pool. If
                # t > t_right, the right, etc.
                dleft_expected += dcontrib_left * contrib_mid * contrib_right
                if ell_mid > 0 and T_mid > 0:
                    dcontrib_mid = fractions.Fraction(
                        _binom(T_mid - 1, ell_mid - 1), ratio_odds ** (ell_mid - 1)
                    )
                    dmid_expected += contrib_left * dcontrib_mid * contrib_right
                if ell_right > 0 and T_right > 0:
                    dcontrib_right = fractions.Fraction(
                        _binom(T_right - 1, ell_right - 1),
                        ratio_odds ** (2 * (ell_right - 1)),
                    )
                    dright_expected += contrib_left * contrib_mid * dcontrib_right

        # determine the appropriate w_high to match our expectation V
        if L:
            w_high = (expected / V) ** (-1 / L)
        else:
            # there is only one way to draw count zero, and it's when no w_t are chosen.
            # in this case, the value of w_high doesn't matter
            w_high = fractions.Fraction(1)

        # adjust w_high until there is no floating-point error converting between
        # rational fractions and floats
        w_high_ = torch.tensor(float(w_high), dtype=dtype).item()
        while w_high_ != float(w_high):
            w_high = fractions.Fraction(w_high_)
            w_high_ = torch.tensor(float(w_high), dtype=dtype).item()

        # adjust our expectations appropriately
        expected *= w_high ** L
        dleft_expected *= w_high ** (L - 1)
        dmid_expected *= w_high ** (L - 1)
        dright_expected *= w_high ** (L - 1)

        w_mid = w_high / ratio_odds
        w_low = w_high / (ratio_odds ** 2)

        w_highs.append(float(w_high))
        w_mids.append(float(w_mid))
        w_lows.append(float(w_low))

        if (
            not opts.method.startswith("log_")
            and float(expected) > torch.finfo(dtype).max
        ):
            raise ValueError(
                "Expected value is too high to be represented in this precision"
            )

        if opts.method.startswith("log_"):
            # go straight into log-odds
            w = torch.cat(
                [
                    torch.full(
                        (T_left,), math.log(w_high), device=opts.device, dtype=dtype
                    ),
                    torch.full(
                        (T_mid,), math.log(w_mid), device=opts.device, dtype=dtype,
                    ),
                    torch.full(
                        (T_right,), math.log(w_low), device=opts.device, dtype=dtype,
                    ),
                ]
            )
        else:
            w = torch.cat(
                [
                    torch.full(
                        (T_left,), float(w_high), device=opts.device, dtype=dtype
                    ),
                    torch.full((T_mid,), float(w_mid), device=opts.device, dtype=dtype),
                    torch.full(
                        (T_right,), float(w_low), device=opts.device, dtype=dtype
                    ),
                ]
            )
        w.requires_grad_(True)

        # permute the odds randomly
        perm = torch.randperm(T, device=opts.device)
        w_permuted = torch.index_select(w, 0, perm)

        actual = func(w_permuted.unsqueeze(0 if batch_first else 1), L, False).squeeze(
            0 if batch_first else 1
        )[-1]

        if L:
            (g,) = torch.autograd.grad(actual, w, torch.ones_like(actual))
        assert not torch.isinf(actual).any()
        if opts.method.startswith("log_"):
            diff = (1.0 - (actual - math.log(expected)).exp()).abs().item()
            if L:
                # rather than computing log (df / dw_t), we've computed
                # d (log f) / d (log w_t).
                # to convert it:
                # log (df / dw_t) = log (
                #               (d (log f) / d (log w_t))
                #               (d (log w_t) / d w_t))
                #               (d (log f) / d f)^-1)
                #                 = log (d (log f) / d (log w_t)) - log w_t + f
                if T_left:
                    dhigh_diff = (
                        (
                            1.0
                            - (
                                g[:t_left].log()
                                - w[:t_left]
                                + actual
                                - math.log(dleft_expected)
                            ).exp()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                if T_mid:
                    dmid_diff = (
                        (
                            1.0
                            - (
                                g[t_left:t_right].log()
                                - w[t_left:t_right]
                                + actual
                                - math.log(dmid_expected)
                            ).exp()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                if T_right:
                    dlow_diff = (
                        (
                            1.0
                            - (
                                g[t_right:].log()
                                - w[t_right:]
                                + actual
                                - math.log(dright_expected)
                            ).exp()
                        )
                        .abs()
                        .max()
                        .item()
                    )
        else:
            diff = ((actual - float(expected)).abs() / float(expected)).item()
            if L:
                if T_left:
                    dhigh_diff = (
                        (g[:t_left] - float(dleft_expected)).abs().max()
                        / float(dleft_expected)
                    ).item()
                if T_mid:
                    dmid_diff = (
                        (g[t_left:t_right] - float(dmid_expected)).abs().max()
                        / float(dmid_expected)
                    ).item()
                if T_right:
                    dlow_diff = (
                        (g[t_right:] - float(dright_expected)).abs().max()
                        / float(dright_expected)
                    ).item()
        diffs.append(diff)
        if L:
            if T_left:
                dhigh_diffs.append(dhigh_diff)
            if T_mid:
                dmid_diffs.append(dmid_diff)
            if T_right:
                dlow_diffs.append(dlow_diff)

    diffs = np.array(diffs)
    dhigh_diffs = np.array(dhigh_diffs if len(dhigh_diffs) else [0.0])
    dmid_diffs = np.array(dmid_diffs if len(dmid_diffs) else [0.0])
    dlow_diffs = np.array(dlow_diffs if len(dlow_diffs) else [0.0])
    w_highs = np.array(w_highs)
    w_mids = np.array(w_mids)
    w_lows = np.array(w_lows)

    print(
        f"log2-repeat={opts.log2_repeat},"
        f"log2-trials={opts.log2_trials},"
        f"log2-highs={opts.log2_highs},"
        f"device={opts.device.type},"
        f"method={opts.method},"
        f"double={opts.double},"
        f"seed={opts.seed},"
        f"log2-ratio-odds={opts.log2_ratio_odds},"
        f"log2-expectation={opts.log2_expectation},"
        f"w-high-mean(std)={w_highs.mean():.4e}({w_highs.std():.4e}),"
        f"w-mid-mean(std)={w_mids.mean():.4e}({w_mids.std():.4e}),"
        f"w-low-mean(std)={w_lows.mean():.4e}({w_lows.std():.4e}),"
        f"nMAE-C(std)={diffs.mean():.4e}({diffs.std():.4e}),"
        f"nMAE-dw-high(std)={dhigh_diffs.mean():.4e}({dhigh_diffs.std():.4e}),"
        f"nMAE-dw-mid(std)={dmid_diffs.mean():.4e}({dmid_diffs.std():.4e}),"
        f"nMAE-dw-low(std)={dlow_diffs.mean():.4e}({dlow_diffs.std():.4e}),"
    )


if __name__ == "__main__":
    main()
