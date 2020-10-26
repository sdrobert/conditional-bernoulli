import torch
import contextlib


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


def R2(w, k_max, keep_hist=False, reverse=False):
    # in w = (*, T)
    # out R = (*, k_max + 1) if keep_hist is False, else
    # R = (*, kmax + 1, T)
    star = w.shape[:-1]
    T = w.shape[-1]
    if reverse:
        w = w.flip(-1)
    w = torch.cat([w, torch.zeros_like(w[..., :1])], -1)  # (*, T + 1)
    w = w.view(-1, 1, T + 1).expand(-1, T + 1, T + 1)  # (N, 1, T + 1, T + 1)
    w = w.tril(-1)  # (N, T + 1, T + 1)
    Rk = torch.ones((w.shape[0], T + 1, 1), device=w.device)  # (N, T + 1, 1)
    if keep_hist:
        hist = [Rk[..., 0]]  # [(N, T)]
    else:
        hist = [Rk[:, -1]]  # [(N, 1)]
    for k in range(k_max):
        Rk = torch.bmm(w, Rk)  # (N, T + 1, 1)
        if keep_hist:
            hist.append(Rk[..., 0])
        else:
            hist.append(Rk[:, -1])
    if keep_hist:
        v = torch.stack(hist, 1).view(*(star + (k_max + 1, T + 1)))
        return v
    else:
        return torch.cat(hist, -1).view(*(star + (k_max + 1,)))


def R3(w, k_max, keep_hist=False, reverse=False):
    # R2 but does block multiplication, avoiding about half the total memory usage
    # in w = (*, T)
    # out R = (*, k_max + 1) if keep_hist is False, else
    # R = (*, kmax + 1, T)
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
    if reverse:
        w = w.flip(-1)
    d = T // 2 + 1
    B_row = w[:, :d]  # (N, d)
    A = B_row.view(-1, 1, d).expand(-1, d, d).tril(-1)  # (N, d, d)
    C = (
        torch.cat([w[:, d:], torch.zeros((w.shape[0], 1), device=w.device)], -1)
        .view(-1, 1, T + 1 - d)
        .expand(-1, T + 1 - d, T + 1 - d)
        .tril(-1)
    )  # (N, T + 1 - d, T + 1 - d)
    Rk_top = torch.ones((w.shape[0], d), device=w.device)  # (N, d)
    Rk_bot = torch.ones((w.shape[0], T + 1 - d), device=w.device)  # (N, T + 1 - d)
    if keep_hist:
        hist = [torch.cat([Rk_top, Rk_bot], -1)]  # [(N, T + 1)]
    else:
        hist = [Rk_bot[:, -1]]  # [(N,)]
    for k in range(k_max):
        Rk_bot = (B_row * Rk_top).sum(1, keepdim=True).expand(
            w.shape[0], T + 1 - d
        ) + torch.bmm(C, Rk_bot.unsqueeze(-1)).squeeze(
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


def main(args=None):
    """Benchmark different count approximations"""
    opts = _parse_args(args)

    if opts.command == "speed":
        return _speed(opts)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--highs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--method", choices=("R", "R2", "R3", "lR"), default="R")

    subparsers = parser.add_subparsers(dest="command", required=True)

    speed_parser = subparsers.add_parser("speed")
    speed_parser.add_argument("--burn-in", type=int, default=100)
    speed_parser.add_argument("--hist", action="store_true", default=False)
    speed_parser.add_argument("--reverse", action="store_true", default=False)

    acc_parser = subparsers.add_parser("acc")
    acc_parser.add_argument("--eps", type=float, default=1e-4)

    return parser.parse_args(args)


def _create_weights(opts, repeat_no=0):
    if opts.seed is not None:
        torch.manual_seed(opts.seed + repeat_no)
    w = torch.randn((opts.trials, opts.batch), device=opts.device).exp()
    if opts.method in {"R2", "R3"}:
        w = w.transpose(0, 1)
    if opts.device.type == "cuda":
        torch.cuda.synchronize()
    return w


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
    import numpy as np

    cum_times = []

    if opts.method == "R":
        _R = R
    elif opts.method == "lR":
        _R = lR
    elif opts.method == "R2":
        _R = R2
    elif opts.method == "R3":
        _R = R3

    if opts.device.type == "cuda":
        timeit = _cuda_timer
    else:
        timeit = _cpu_timer

    w = _create_weights(opts)

    for _ in range(opts.burn_in):
        _R(w, opts.highs, opts.hist, opts.reverse)

    for repeat_no in range(opts.repeat):

        with timeit() as timer:
            _R(w, opts.highs, opts.hist, opts.reverse)

        cum_times.append(timer())

    cum_times = np.array(cum_times)

    print(
        "batch={},burn_in={},command={},device={},highs={},hist={},method={},"
        "repeat={},reverse={},seed={},trials={},mu(ms)={:.02f},std(ms)={:.02f}"
        "".format(
            opts.batch,
            opts.burn_in,
            opts.command,
            opts.device.type,
            opts.highs,
            opts.hist,
            opts.method,
            opts.repeat,
            opts.reverse,
            opts.seed,
            opts.trials,
            cum_times.mean(),
            cum_times.std(),
        )
    )


if __name__ == "__main__":
    main()
