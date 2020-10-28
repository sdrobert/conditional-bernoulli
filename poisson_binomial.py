import torch
import contextlib
import math
import argparse
import numpy as np
import fractions

# for log count functions, we want -inf to behave the same way as if it were zero in
# the regular count function. Thus we replace any occurrence with a log value that,
# when exponentiated, is nonzero, but only just. The division by two here ensures that
# we can add two EPS_INF values without the result being zero.
EPS_INF = math.log(torch.finfo(torch.float32).tiny) / 2


def R(w, k_max, keep_hist=False, reverse=False):
    # in w = (T, *)
    # out R = (k_max + 1, *) if keep_hist is False, otherwise
    # R = (T + 1, k_max + 1, *)
    R0 = torch.ones_like(w[:1, ...])
    Rrest = torch.zeros((k_max,) + w[0].size(), device=w.device, dtype=w.dtype)
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


def R2(w, k_max, keep_hist=False, reverse=False):
    # in w = (*, T)
    # out R = (*, k_max + 1) if keep_hist is False, else
    # R = (*, kmax + 1, T + 1)
    star = w.shape[:-1]
    T = w.shape[-1]
    if reverse:
        w = w.flip(-1)
    w = torch.cat([w, torch.zeros_like(w[..., :1])], -1)  # (*, T + 1)
    w = w.view(-1, 1, T + 1).expand(-1, T + 1, T + 1)  # (N, 1, T + 1, T + 1)
    N = w.shape[0]
    w = w.tril(-1)  # (N, T + 1, T + 1)
    Rk = torch.ones((N, T + 1, 1), device=w.device, dtype=w.dtype)
    if keep_hist:
        hist = [Rk[..., 0]]  # [(N, T + 1)]
    else:
        hist = [Rk[:, -1]]  # [(N, 1)]
    for _ in range(k_max):
        Rk = torch.bmm(w, Rk)  # (N, T + 1, 1)
        if keep_hist:
            hist.append(Rk[..., 0])
        else:
            hist.append(Rk[:, -1])
    if keep_hist:
        return torch.stack(hist, 1).view(*(star + (k_max + 1, T + 1)))
    else:
        return torch.cat(hist, -1).view(*(star + (k_max + 1,)))


def R3(w, k_max, keep_hist=False, reverse=False):
    # R2 but does block multiplication, avoiding about half the total memory usage
    # in w = (*, T)
    # out R = (*, k_max + 1) if keep_hist is False, else
    # R = (*, kmax + 1, T + 1)
    #
    #     d           d
    #     |           |
    #     v           v
    # 0 0 0 0 0     0 0 0 0
    # a 0 0 0 0     a 0 0 0
    # a a 0 0 0     b b 0 0
    # b b c 0 0     b b c 0
    # b b c c 0
    #
    # - the rows in B are the same, so we don't waste memory on them
    star = w.shape[:-1]
    T = w.shape[-1]
    w = w.view(-1, T)
    N = w.shape[0]
    if reverse:
        w = w.flip(-1)
    d = T // 2 + 1
    B_row = w[:, :d]  # (N, d)
    A = B_row.view(-1, 1, d).expand(-1, d, d).tril(-1)  # (N, d, d)
    C = (
        torch.cat([w[:, d:], torch.zeros((N, 1), device=w.device, dtype=w.dtype)], -1)
        .view(-1, 1, T + 1 - d)
        .expand(-1, T + 1 - d, T + 1 - d)
        .tril(-1)
    )  # (N, T + 1 - d, T + 1 - d)
    Rk_top = torch.ones((N, d), device=w.device, dtype=w.dtype)  # (N, d)
    Rk_bot = torch.ones(
        (N, T + 1 - d), device=w.device, dtype=w.dtype
    )  # (N, T + 1 - d)
    if keep_hist:
        hist = [torch.cat([Rk_top, Rk_bot], -1)]  # [(N, T + 1)]
    else:
        hist = [Rk_bot[:, -1]]  # [(N,)]
    for _ in range(k_max):
        Rk_bot = (B_row * Rk_top).sum(1, keepdim=True).expand(N, T + 1 - d) + torch.bmm(
            C, Rk_bot.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (N, T + 1 - d)
        Rk_top = torch.bmm(A, Rk_top.unsqueeze(-1)).squeeze(-1)  # (N, d)
        if keep_hist:
            hist.append(torch.cat([Rk_top, Rk_bot], -1))  # [(N, T + 1)]
        else:
            hist.append(Rk_bot[:, -1])  # [(N,)]
    if keep_hist:
        return torch.stack(hist, 1).view(*(star + (k_max + 1, T + 1)))
    else:
        return torch.stack(hist, -1).view(*(star + (k_max + 1,)))


def R4(w, k_max, keep_hist=False, reverse=False):
    # similar to R2 and R3, but never expresses the full matrix, instead using
    # cumsum
    star = w.shape[:-1]
    T = w.shape[-1]
    w = w.view(-1, T)
    N = w.shape[0]
    if reverse:
        w = w.flip(-1)
    # w = torch.cat([w, torch.zeros_like(w[..., :1])], -1)  # (N, T + 1)
    Rk = torch.ones((N, T + 1), device=w.device, dtype=w.dtype)
    zeros = torch.zeros((N, 1), device=w.device, dtype=w.dtype)
    if keep_hist:
        hist = [Rk]  # (N, T + 1)
    else:
        hist = [Rk[:, -1]]  # (N,)
    for _ in range(k_max):
        Rk = torch.cat([zeros, torch.cumsum(w * Rk[:, :-1], 1)], -1)
        if keep_hist:
            hist.append(Rk)
        else:
            hist.append(Rk[:, -1])
    if keep_hist:
        return torch.stack(hist, 1).view(*(star + (k_max + 1, T + 1)))
    else:
        return torch.stack(hist, -1).view(*(star + (k_max + 1,)))


def R5(w, k_max, keep_hist=False, reverse=False):
    # Chen's 1994 method from Theorem 3
    # w = (*, n)
    # out = (*, k_max + 1)
    star = w.shape[:-1]
    n = w.shape[-1]  # normally this is T, but Chen uses this as below
    w = w.view(-1, n)

    # First calculate all the T(i, w) terms.
    T = [(w ** i).sum(1) for i in range(1, k_max + 1)]

    v = _R5(w, k_max, T)  # (N, k_max + 1)

    if keep_hist:
        # Chen's method produces R(L, w) for *only* the full set w. If we want more,
        # we have to re-calculate the recursion. However, as per Chen, we can avoid some
        # re-computation if we subtract terms from T
        v = [v]
        for _ in range(n - 1, -1, -1):
            if reverse:
                # cut the first weight when calculating R(L, w \ {w_t})
                w_t = w[:, 0]
                w = w[:, 1:]
            else:
                # cut the last weight when calculating R(L, w \ {w_t})
                w_t = w[:, -1]
                w = w[:, :-1]
            T = [T[i - 1] - w_t ** i for i in range(1, k_max + 1)]
            v.insert(0, _R5(w, k_max, T))
        v = torch.stack(v, -1).view(*(star + (k_max + 1, n + 1)))
    else:
        v = v.view(*(star + (k_max + 1,)))

    return v


def _R5(w, k_max, T):
    N = w.shape[0]
    R = [torch.ones((N,), device=w.device, dtype=w.dtype)]
    zeros = torch.zeros((N,), device=w.device, dtype=w.dtype)
    for k_prime in range(1, k_max + 1):
        Rk_prime = zeros
        for i in range(1, k_prime + 1):
            if i % 2:
                Rk_prime = Rk_prime + T[i - 1] * R[k_prime - i]
            else:
                Rk_prime = Rk_prime - T[i - 1] * R[k_prime - i]
        Rk_prime = Rk_prime / k_prime
        R.append(Rk_prime)
    return torch.stack(R, -1)  # (N, k + 1)


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


def lR(logits, k_max, keep_hist=False, reverse=False):
    # in logits = (T, *)
    # out log_R = (k_max + 1, *)
    log_R0 = torch.zeros_like(logits[:1, ...])
    log_Rrest = torch.full(
        (k_max,) + logits[0].size(), EPS_INF, device=logits.device, dtype=logits.dtype,
    )
    logits = logits.unsqueeze(1)
    if keep_hist:
        hist = [torch.cat([log_R0, log_Rrest])]
    T = logits.shape[0]
    for t in range(T):
        if reverse:
            t = T - t - 1
        x = torch.cat([log_R0, log_Rrest[:-1]], 0)
        x = x + logits[t]
        if k_max:
            log_Rrest = _logaddexp(log_Rrest, x)
        else:
            log_Rrest = x
        del x
        if keep_hist:
            hist.append(torch.cat([log_R0, log_Rrest]))
    if keep_hist:
        return torch.stack(hist, 0)
    else:
        return torch.cat([log_R0, log_Rrest])


def lR4(logits, k_max, keep_hist=False, reverse=False):
    # in logits = (*, T)
    # out log_R = (*, k_max + 1)
    star = logits.shape[:-1]
    T = logits.shape[-1]
    logits = logits.view(-1, T)
    logits = logits.masked_fill(torch.isinf(logits), EPS_INF)
    N = logits.shape[0]
    if reverse:
        logits = logits.flip(-1)
    lRk = torch.zeros((N, T + 1), device=logits.device, dtype=logits.dtype)
    ninfs = torch.full((N, 1), EPS_INF, device=logits.device, dtype=logits.dtype)
    if keep_hist:
        hist = [lRk]  # (N, T + 1)
    else:
        hist = [lRk[:, -1]]  # (N,)
    for _ in range(k_max):
        x = logits + lRk[:, :-1]
        x = x.masked_fill(torch.isinf(x), EPS_INF)
        lRk = torch.cat([ninfs, _logcumsumexp(x, 1)], -1)
        if keep_hist:
            hist.append(lRk)
        else:
            hist.append(lRk[:, -1])
    if keep_hist:
        return torch.stack(hist, 1).view(*(star + (k_max + 1, T + 1)))
    else:
        return torch.stack(hist, -1).view(*(star + (k_max + 1,)))


def lR5(logits, k_max, keep_hist=False, reverse=False):
    # in logits = (*, T)
    # out log_R = (*, k_max + 1)
    star = logits.shape[:-1]
    n = logits.shape[-1]
    logits = logits.view(-1, n).clamp(min=EPS_INF)

    T = [(i * logits).logsumexp(1) for i in range(1, k_max + 1)]

    v = _lR5(k_max, T)

    if keep_hist:
        v = [v]
        for _ in range(n - 1, -1, -1):
            if reverse:
                logits_t = logits[:, 0]
                logits = logits[:, 1:]
            else:
                logits_t = logits[:, -1]
                logits = logits[:, :-1]
            T = [_logsubexp(T[i - 1], i * logits_t) for i in range(1, k_max + 1)]
            v.insert(0, _lR5(k_max, T))
        v = torch.stack(v, -1).view(*(star + (k_max + 1, n + 1)))
    else:
        v = v.view(*(star + (k_max + 1,)))

    return v


def _lR5(k_max, T):
    N = T[0].shape[0]
    lR = [torch.zeros((N,), device=T[0].device, dtype=T[0].dtype)]
    ninfs = torch.full((N,), -float("inf"), device=T[0].device, dtype=T[0].dtype)
    for k_prime in range(1, k_max + 1):
        lRk_prime = ninfs
        for i in range(1, k_prime + 1):
            if i % 2:
                lRk_prime = _logaddexp(lRk_prime, T[i - 1] + lR[k_prime - i])
            else:
                lRk_prime = _logsubexp(lRk_prime, T[i - 1] + lR[k_prime - i])
        lRk_prime = lRk_prime - math.log(k_prime)
        lR.append(lRk_prime)
    return torch.stack(lR, -1)  # (N, k + 1)


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


def main(args=None):
    """Benchmark different count approximations"""
    opts = _parse_args(args)

    if opts.command == "speed":
        return _speed(opts)
    elif opts.command == "acc":
        return _accuracy(opts)


def _parse_args(args):
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--highs", type=int, default=10)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument(
        "--method",
        choices=("R", "R2", "R3", "R4", "lR", "lR4", "R5", "lR5"),
        default="R",
    )
    parser.add_argument("--double", action="store_true", default=False)

    subparsers = parser.add_subparsers(dest="command", required=True)

    speed_parser = subparsers.add_parser("speed")
    speed_parser.add_argument("--burn-in", type=int, default=100)
    speed_parser.add_argument("--hist", action="store_true", default=False)
    speed_parser.add_argument("--reverse", action="store_true", default=False)
    speed_parser.add_argument("--batch", type=int, default=1)

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

    if opts.method == "R":
        _R = R
    elif opts.method == "lR":
        _R = lR
    elif opts.method == "R2":
        _R = R2
    elif opts.method == "R3":
        _R = R3
    elif opts.method == "R4":
        _R = R4
    elif opts.method == "lR4":
        _R = lR4
    elif opts.method == "R5":
        _R = R5
    elif opts.method == "lR5":
        _R = lR5

    if opts.device.type == "cuda":
        timeit = _cuda_timer
    else:
        timeit = _cpu_timer

    dtype = torch.float64 if opts.double else torch.float32

    w = torch.zeros(opts.trials, opts.batch, device=opts.device, dtype=dtype)
    if opts.method in {"R2", "R3", "R4", "lR4", "R5", "lR5"}:
        w = w.transpose(0, 1)
    if opts.method.startswith("l"):
        w = w.log()
    if opts.device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(opts.burn_in):
        _R(w, opts.highs, opts.hist, opts.reverse)

    for repeat_no in range(opts.repeat):

        with timeit() as timer:
            v = _R(w, opts.highs, opts.hist, opts.reverse)
        assert not torch.isnan(v).any() and not torch.isinf(v).any()

        cum_times.append(timer())

    cum_times = np.array(cum_times)

    print(
        "repeat={},trials={},highs={},device={},method={},double={},burn-in={},"
        "hist={},reverse={},batch={},mean-ms={:.4f}({:.4f})".format(
            opts.repeat,
            opts.trials,
            opts.highs,
            opts.device.type,
            opts.method,
            opts.double,
            opts.burn_in,
            opts.hist,
            opts.reverse,
            opts.batch,
            cum_times.mean(),
            cum_times.std(),
        )
    )


def _binom(T, L):
    return math.factorial(T) // (math.factorial(T - L) * math.factorial(L))


def _accuracy(opts):
    diffs = []
    dw_diffs = []
    daw_diffs = []

    if opts.method == "R":
        _R = R
    elif opts.method == "lR":
        _R = lR
    elif opts.method == "R2":
        _R = R2
    elif opts.method == "R3":
        _R = R3
    elif opts.method == "R4":
        _R = R4
    elif opts.method == "lR4":
        _R = lR4
    elif opts.method == "R5":
        _R = R5
    elif opts.method == "lR5":
        _R = lR5

    dtype = torch.float64 if opts.double else torch.float32

    ratio_odds = 2 ** opts.log2_ratio_odds

    # this method is similar to that of
    # https://doi.org/10.1016/j.csda.2012.10.006
    # but is essentially prop 1.c of 10.2307/2337119, a generalization of Vandermonde's
    # identity.
    # the intuition behind the identity is: if you split your odds into two mutually
    # exclusive subsets, w.l.o.g. w_{<=t} and w_{> t}, then the number of ways you can
    # chooose L from them is the sum over k of the number of ways to choose k from
    # w_{<=t} and L - k from w_{>t}. If all weights in w_{<=t} are identically w',
    # T-choose-k of that subset is simply binom(t, k) * w'^k. Setting the w_{> t}
    # identically to aw' gives binom(T - t, L - k) * w'^{L - k} a^{L - k}

    for repeat_no in range(opts.repeat):
        if opts.seed is not None:
            torch.manual_seed(opts.seed + repeat_no)

        # choose some number of odds t to have value w', the rest
        # w' / ratio_odds = aw'
        t = torch.randint(1, opts.trials, (1,)).item()

        # determine the expected value assuming w' == 1
        expected = fractions.Fraction(0)
        dw_prime_expected = fractions.Fraction(0)
        daw_prime_expected = fractions.Fraction(0)
        for k in range(max(0, t - opts.trials + opts.highs), min(opts.highs, t) + 1):
            contrib_aw = fractions.Fraction(
                _binom(opts.trials - t, opts.highs - k), ratio_odds ** (opts.highs - k),
            )
            contrib_w = _binom(t, k)
            expected += contrib_aw * contrib_w
            # the derivative w.r.t. w_t essentially fixes the choice of t and sets
            # its weight to 1. If w_t = w', this fixes one choice from the w' block.
            # Similarly, if w_t = aw', it fixes one choice from the aw' block.
            # It's important to remember that the w_t are all supposed to be different
            # variables, even if their values are the same.
            if k > 0 and t > 0:
                dw_prime_expected += _binom(t - 1, k - 1) * contrib_aw
            if opts.highs > k and opts.trials > t:
                daw_prime_expected += (
                    fractions.Fraction(
                        _binom(opts.trials - t - 1, opts.highs - k - 1),
                        ratio_odds ** (opts.highs - k - 1),
                    )
                    * contrib_w
                )

        # determine some w' such that the expectation would be opts.expectation.
        # exp[count(w, L)] == w'^L exp[count(1, L)]
        # v == w'^L exp[count(1, L)]
        # w == exp[count(1, L)]^{-1/L}
        if opts.highs:
            w_prime = (expected / (2 ** opts.log2_expectation)) ** (-1 / opts.highs)
        else:
            # there is only one way to draw count zero, and it's when no w_t are chosen.
            # in this case, the value of w_prime doesn't matter
            w_prime = fractions.Fraction(torch.rand(1).item())

        # adjust w_prime until there is no floating-point error converting between
        # rational fractions and floats
        w_prime_ = torch.tensor(float(w_prime), dtype=dtype).item()
        while w_prime_ != float(w_prime):
            w_prime = fractions.Fraction(w_prime_)
            w_prime_ = torch.tensor(float(w_prime), dtype=dtype).item()

        # adjust our expectations appropriately
        expected *= w_prime ** opts.highs
        dw_prime_expected *= w_prime ** (opts.highs - 1)
        daw_prime_expected *= w_prime ** (opts.highs - 1)

        aw_prime = w_prime / ratio_odds

        if not opts.method.startswith("l") and float(expected) > torch.finfo(dtype).max:
            raise ValueError(
                "Expected value is too high to be represented in this precision"
            )

        if opts.method.startswith("l"):
            # go straight into log-odds
            w = torch.cat(
                [
                    torch.full(
                        (t,), math.log(w_prime), device=opts.device, dtype=dtype
                    ),
                    torch.full(
                        (opts.trials - t,),
                        math.log(aw_prime),
                        device=opts.device,
                        dtype=dtype,
                    ),
                ]
            )
        else:
            w = torch.cat(
                [
                    torch.full((t,), float(w_prime), device=opts.device, dtype=dtype),
                    torch.full(
                        (opts.trials - t,),
                        float(aw_prime),
                        device=opts.device,
                        dtype=dtype,
                    ),
                ]
            )
        w.requires_grad_(True)

        # permute the odds randomly
        perm = torch.randperm(opts.trials, device=opts.device)
        w_permuted = torch.index_select(w, 0, perm)

        actual = _R(w_permuted, opts.highs)[-1]

        if opts.highs:
            (g,) = torch.autograd.grad(actual, w, torch.ones_like(actual))
        assert not torch.isinf(actual).any()
        if opts.method.startswith("l"):
            err = (1.0 - (actual - math.log(expected)).exp()).abs().item()
            if opts.highs:
                # rather than computing log (df / dw_t), we've computed
                # d (log f) / d (log w_t).
                # to convert it:
                # log (df / dw_t) = log (
                #               (d (log f) / d (log w_t))
                #               (d (log w_t) / d w_t))
                #               (d (log f) / d f)^-1)
                #                 = log (d (log f) / d (log w_t)) - log w_t + f
                dw_err = (
                    (
                        1.0
                        - (
                            g[:t].log() - w[:t] + actual - math.log(dw_prime_expected)
                        ).exp()
                    )
                    .abs()
                    .max()
                    .item()
                )
                daw_err = (
                    (
                        1.0
                        - (
                            g[t:].log() - w[t:] + actual - math.log(daw_prime_expected)
                        ).exp()
                    )
                    .abs()
                    .max()
                    .item()
                )
        else:
            err = ((actual - float(expected)).abs() / float(expected)).item()
            if opts.highs:
                dw_err = (
                    (g[:t] - float(dw_prime_expected)).abs().max()
                    / float(dw_prime_expected)
                ).item()
                daw_err = (
                    (g[t:] - float(daw_prime_expected)).abs().max()
                    / float(daw_prime_expected)
                ).item()
        diffs.append(err)
        if opts.highs:
            dw_diffs.append(dw_err)
            daw_diffs.append(daw_err)

    diffs = np.array(diffs)
    dw_diffs = np.array(dw_diffs if len(dw_diffs) else [0.0])
    daw_diffs = np.array(daw_diffs if len(daw_diffs) else [0.0])

    print(
        "repeat={},trials={},highs={},device={},method={},double={},seed={},"
        "log2-ratio-odds={},log2-expectation={},MAE={:.2e}({:.2e}),"
        "dw-MAE={:.2e}({:.2e}),daw-MAE={:.2e}({:.2e})".format(
            opts.repeat,
            opts.trials,
            opts.highs,
            opts.device,
            opts.method,
            opts.double,
            opts.seed,
            opts.log2_ratio_odds,
            opts.log2_expectation,
            diffs.mean(),
            diffs.std(),
            dw_diffs.mean(),
            dw_diffs.std(),
            daw_diffs.mean(),
            daw_diffs.std(),
        )
    )


if __name__ == "__main__":
    main()
