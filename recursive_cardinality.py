# Comparing implementations of the generalized binomial function.
#
# For a set of weights x, the generalized binomial function sums the products of weights
# of all subsets x of some cardinality, i.e.
#
#   gbinom(x, c) = sum_{X : |X| = c} prod_{x in X} x
#
# Letting w be a set of odds and L be some number of highs, the quantity gbinom(w, L) is
# the denominator of an associated Conditional Bernoulli density.
#
# There are a few ways of computing gbinom(x, c). "naive", "fft1", and "fft2" are based
# on the message-passing algorithm proposed in
#
#   Tarlow et al. (2012) Fast exact inference for Recursive Cardinality models.
#
# whereas "cum" and "lcum" are based on the recurrence defined in
#
#   Howard. (1972) Discussion on Professor Cox's paper.
#
# Descriptions of each algorithm can be found at the beginning of their functions
# (pytorch ver.). They all return a vector [gbinom(x, 0), gbinom(x, 1), ..., gbinom(x,
# T)], where T is the number of elements in x.
#
# Calling this script from command line
#
#   python recursive_cardinality.py
#
# will run a mini-benchmark of the algorithms. Comments at the beginning of the
# "main" function describe the benchmark as well as some results.

import torch
import math
import numpy as np
import time
import warnings


@torch.no_grad()
@torch.jit.script
def naive(x: torch.Tensor) -> torch.Tensor:
    # A naive implementation of the message passing algorithm proposed by Tarlow et al.
    #
    # The algorithm constructs a binary tree of latent variables z. The T leaves z_t
    # have binary values and associated factors:
    #
    #   f(z_t = 0) = 1, f(z_t = 1) = x_t.
    #
    # Internal nodes z_n with left and right children z_l and z_r are assigned the
    # indicator factor f(z_n|z_l,z_r) = I[z_n = z_l + z_r]. The f(z_n=t) can be
    # marginalized (upwards) as
    #
    #   f(z_n = t) = sum_tau f(z_l = tau)f(z_r = t - tau)
    #
    # which is a linear convolution. If z_{root} is the root node of the tree, then
    #
    #   f(z_{root} = t) = gbinom(x, t).
    #
    # Though "naive", this algorithm still performs batched computation of all upward
    # messages at a given level. x is first converted to shape (T, 2) where
    #
    #   x[t, v] = f(z_t = v).
    #
    # Each iteration of the outer loop splits x into half along the first dimension
    # to represent the left and right children of some parent. A linear convolution
    # is performed to marginalize the parents. The parents can have twice as many
    # values as the children but there are only half as many parents.
    #
    # This algorithm is "naive" because it performs the convolution using nested loops.

    rest = x.shape[:-1]
    T = x.size(-1)
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT  # restrict |x| to be some power of 2 for simplicity.
    x = x.unsqueeze(-1)
    x = torch.cat([torch.ones_like(x), x], -1)  # (..., T, 2): the leaves
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        x_l, x_r = x[..., :Nout, :], x[..., Nout:, :]  # left and right children
        x = torch.zeros(rest + (Nout, Tout), dtype=x.dtype)  # parent
        for t in range(Tout):
            for tau in range(max(t - Tin + 1, 0), min(Tin, t + 1)):
                x[..., t] += x_l[..., tau] * x_r[..., t - tau]
    return x[..., 0, : T + 1]


def naive_np(x: np.ndarray) -> np.ndarray:
    # Numpy version of "naive"
    rest = x.shape[:-1]
    T = x.shape[-1]
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT
    x = x[..., None]
    x = np.concatenate([np.ones_like(x), x], axis=-1)  # (..., T, 2)
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        x_1, x_2 = x[..., :Nout, :], x[..., Nout:, :]
        x = np.zeros(rest + (Nout, Tout), dtype=x.dtype)
        for t in range(Tout):
            for tau in range(max(t - Tin + 1, 0), min(Tin, t + 1)):
                x[..., t] += x_1[..., tau] * x_2[..., t - tau]
    return x[..., 0, : T + 1]


@torch.no_grad()
@torch.jit.script
def cum(x: torch.Tensor) -> torch.Tensor:
    # Implementation of Howard's recurrence using a cumulative sum operator.
    #
    # This method takes advantage of the identity
    #
    #   gbinom(x, ell) = gbinom(x \ {x_t}, ell) + x_t gbinom(x \ {x_t}, ell - 1)
    #
    # to construct a table with elements r[ell, t] = gbinom({x_1, ... x_t}, ell). The
    # return value is the final column r[:, T]. The algorithm computes the nonzero part
    # of the next row r[t, t + 1:] with cumulative sum of the product of the previous
    # row r[t - 1, :-1] with the values x[t:].
    rest = x.shape[:-1]
    T = x.size(-1)
    r = torch.ones(rest + (T + 1,), dtype=x.dtype)
    ret = torch.empty(rest + (T + 1,), dtype=x.dtype)
    ret[..., 0] = 1
    for t in range(T):
        r = torch.cumsum(x[..., t:] * r[..., :-1], -1)
        ret[..., t + 1] = r[..., -1]
    return ret


def cum_np(x: np.ndarray) -> np.ndarray:
    # Numpy version of "cum"
    rest = x.shape[:-1]
    T = x.shape[-1]
    r = np.ones(rest + (T + 1,), dtype=x.dtype)
    ret = np.empty(rest + (T + 1,), dtype=x.dtype)
    ret[..., 0] = 1
    for t in range(T):
        r = np.cumsum(x[..., t:] * r[..., :-1], -1)
        ret[..., t + 1] = r[..., -1]
    return ret


@torch.no_grad()
@torch.jit.script
def lcum(lx: torch.Tensor) -> torch.Tensor:
    # log-domain version of "cum"
    #
    # "lx[t] = log x[t]". "ret[t] = log gbinom(x, t)". Uses torch.logcumsumexp instead
    # of cumsum.
    rest = lx.shape[:-1]
    T = lx.size(-1)
    lr = torch.zeros(rest + (T + 1,), dtype=lx.dtype)
    ret = torch.empty(rest + (T + 1,), dtype=lx.dtype)
    ret[..., 0] = 0
    for t in range(T):
        lr = torch.logcumsumexp(lx[..., t:] + lr[..., :-1], -1)
        ret[..., t + 1] = lr[..., -1]
    return ret


def lcum_np(lx: torch.Tensor) -> torch.Tensor:
    # Numpy version of "lcum".
    #
    # Note that this isn't as numerically stable as lcum_np. There isn't a logcumsumexp
    # in Numpy. logcumsumexp in torch performs logsumexp on each pair of successive
    # elements. Here we simply subtract the maximum element in the row to add back
    # later.
    rest = lx.shape[:-1]
    T = lx.shape[-1]
    lr = np.zeros(rest + (T + 1,), dtype=lx.dtype)
    ret = np.empty(rest + (T + 1,), dtype=lx.dtype)
    ret[..., 0] = 0
    for t in range(T):
        lr = lx[..., t:] + lr[..., :-1]
        max_ = np.max(lr, -1, keepdims=True)
        lr = np.log(np.cumsum(np.exp(lr - max_), -1)) + max_
        ret[..., t + 1] = lr[..., -1]
    return ret


@torch.no_grad()
@torch.jit.script
def fft1(x: torch.Tensor) -> torch.Tensor:
    # FFT convolution version of "naive" with intermediate FFT operations
    #
    # Identical to "naive" except for the way that parent values are computed. First, an
    # RFFT is performed on the sequence of values for a given node. By the Convolution
    # Theorem, the Fourier Transform of the parent's values is equal to the product of
    # the FTs of its children. Then the transform is inverted to get the parent's
    # values.
    T = x.size(-1)
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT
    x = x.unsqueeze(-1)
    x = torch.cat([torch.ones_like(x), x], -1)
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        X = torch.fft.rfft(x, n=Tout)
        X = X[..., :Nout, :] * X[..., Nout:, :]
        x = torch.fft.irfft(X, n=Tout)
        assert x.shape == (Nout, Tout)
    return x[..., 0, : T + 1]


def fft1_np(x: np.ndarray) -> np.ndarray:
    # Numpy version of "fft1"
    T = x.shape[-1]
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT
    x = x[..., None]
    x = np.concatenate([np.ones_like(x), x], -1)
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        X = np.fft.rfft(x, n=Tout)
        X = X[..., :Nout, :] * X[..., Nout:, :]
        x = np.fft.irfft(X, n=Tout)
        assert x.shape == (Nout, Tout)
    return x[..., 0, : T + 1]


@torch.no_grad()
@torch.jit.script
def fft2(x: torch.Tensor) -> torch.Tensor:
    # FFT convolution version of "naive" without intermediate FFT operations
    #
    # Like "fft1", it performs convolutions via FFTs. Instead of inverting the FFT
    # after each step, this version performs one RFFT for all the leaves and one iRFFT
    # at the root. While avoiding intermediate RFFT/iRFFT operations, the initial RFFT
    # requires each leaf's values to be right-padded with zeros in order to avoid the
    # effects of circular convolution.
    T = x.size(-1)
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT
    x = x.unsqueeze(-1)
    x = torch.cat([torch.ones_like(x), x], -1)
    X = torch.fft.rfft(x, n=2 * T)
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        X = X[..., :Nout, :] * X[..., :Nout, :]
    x = torch.fft.irfft(X, n=2 * T)
    return x[..., 0, : T + 1]


def fft2_np(x: np.ndarray) -> np.ndarray:
    # Numpy version of "fft2"
    T = x.shape[-1]
    logT_ = math.log(T) / math.log(2)
    logT = int(logT_)
    assert logT_ == logT
    x = x[..., None].astype(np.float64)
    x = np.concatenate([np.ones_like(x), x], -1)
    X = np.fft.rfft(x, n=2 * T)
    Tout, Nout = 2, T
    for _ in range(1, logT + 1):
        Tin, Nin = Tout, Nout
        Tout, Nout = 2 * Tin, Nin // 2
        X = X[..., :Nout, :] * X[..., :Nout, :]
    x = np.fft.irfft(X, n=2 * T)
    return x[..., 0, : T + 1]


def binom(n: int, k: int) -> int:
    return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))


def main():
    # A simple benchmark to compare implementations of gbinom on the CPU
    #
    # gbinom(x, t) reduces to binom(T, t) if x is uniformly 1. x_t = 1 is a plausible
    # value for odds in the Conditional Bernoulli as it corresponds to P(b_t = 1) = 0.5.
    # This benchmark measures both the speed and accuracy of each above algorithm
    # againsts this special case. It prints one line for earch of combination of the
    # following settings:
    #
    # logT : |x| = 2 ** logT = T is the number of elements in x.
    # precision: whether results and intermediary values use single (32-bit) or double
    #            (64-bit) computations
    # backend : whether the algorithm was written for pytorch or numpy.
    # style: one of the above algorithms: "naive", "cum", "lcum", "fft1", or "fft2"
    #
    # Runtimes are in seconds, averaged over some number of repeated runs. Algorithm
    # error per partition is calculated as
    #
    #    err[p] = sum_t abs(expected_i - actual_i) / partition_size[p].
    #
    # By default, error rates are calculated over quadrants.
    #
    # Some notes on the algorithms and results:
    #
    # - The greatest return value is in the middle of the vector; the least values in
    #   the periphery. binom(T, T // 2) is approximately 2 ** (logT - 1). In contrast,
    #   binom(T, 0) = 1. The magnitude of entries are thus very different, making any
    #   algorithm involving floating-point arithmetic susceptible to error. This is also
    #   one of the reasons we partition error rates.
    # - The runtimes with 32-bit and 64-bit precision are most likely comparable on the
    #   CPU since most modern CPUs perform 64-bit arithmetic. 32-bit operations are
    #   included for extrapolating accuracy to CUDA, which much prefers the shorter
    #   floats.
    # - fft1 has the best runtime complexity. At higher T, it runs consistently faster
    #   than the other algorithms. At middling T values, fft2 can outperform fft1 by
    #   skipping the additional FFT/iFFT operations. However, that benefit disappears
    #   as T gets larger and the wasted computation and memory of the larger initial
    #   FFT catches up to it. "fft1" is about 2-10x faster than "cum", "cum" 2-10x
    #   faster than "lcum", and "lcum" around 100x faster than "naive". I disabled
    #   "naive" when T > 128.
    # - Accuracy is harder to analyze. In general, "naive" and "cum" beat out the
    #   FFT-based algorithms. Whereas the error is roughly equal in each quadrant of the
    #   FFT algorithms, the error in "naive", "cum", and "lcum" is much higher in the
    #   middle two quadrants than in the first and last. Since error is calculated as an
    #   absolute difference and the expected values around T // 2 are highest, this
    #   difference in error by quadrants is expected. Explaining why it did not happen
    #   for the FFT algorithms is trickier. The left and right children of the binary
    #   tree ought always have the same density over values since x is uniformly 1. This
    #   implies multiplication in the fourier domain is pointwise stable (there are no
    #   differences in magnitude between multiples). The problem is therefore likely a
    #   function of the RFFT/iFFT algorithms. I've heard mention of catastrophic
    #   cancellation in FFTs online, but need to look into it further. I suspect the
    #   uneven distribution of errors will matter most when events are sparse with
    #   respect to T.
    # - "lcum" is less stable than "cum". "lcum" tends to do about the same or worse
    #   than "fft1" for the middle quadrants and much better than it on the first and
    #   last, but is 10-100x less accurate on than "cum" and "naive" in general. The
    #   inaccuracy is likely due to the repeated logsumexp operations necessary to sum
    #   terms. Any benefits "lcum" would receive over other methods for accurate
    #   multiplication is mitigated by the fact that x is uniformly 1. However, "lcum"
    #   is the only algorithm which reliably does not overflow when using 32-bit
    #   precision. This is less a stability issue and more one of representation:
    #   binom(n, k) quickly exceeds the maximum single-precision float. "cum" seems
    #   fine at double precision.
    warnings.simplefilter("ignore")  # disable some warnings for when we overflow
    trials = 2 ** 5  # the number of trials to average the time of each method over
    max_logT = 10  # the maximum power of 2 to calculate
    partitions = 4  # the number of partitions of the return value to divide error rate
    #                 computations into
    methods = {
        ("naive", "torch"): (naive, False),
        ("naive", "numpy"): (naive_np, False),
        ("cum", "torch"): (cum, False),
        ("cum", "numpy"): (cum_np, False),
        ("lcum", "torch"): (lcum, True),
        ("lcum", "numpy"): (lcum_np, True),
        ("fft1", "torch"): (fft1, False),
        ("fft1", "numpy"): (fft1_np, False),
        ("fft2", "torch"): (fft2, False),
        ("fft2", "numpy"): (fft2_np, False),
    }
    for logT in range(max_logT + 1):
        T = 2 ** logT
        # we don't turn the expected value into a numpy array b/c python integer
        # precision can go really high
        exp = [binom(T, k) for k in range(T + 1)]
        for precision in ("float", "double"):
            for backend in ("numpy", "torch"):
                if backend == "numpy":
                    dtype = np.float32 if precision == "float" else np.float64
                    in_ = np.ones(T, dtype=dtype)
                    log_in = np.zeros(T, dtype=dtype)
                else:
                    dtype = torch.float32 if precision == "float" else torch.float64
                    in_ = torch.ones(T, dtype=dtype)
                    log_in = torch.zeros(T, dtype=dtype)
                for style in ("naive", "cum", "lcum", "fft1", "fft2"):
                    if style in {"naive"} and logT > 7:
                        continue
                    method, act_is_log = methods[(style, backend)]
                    start = time.time()
                    for _ in range(trials):
                        method(in_)
                    end = time.time()
                    avg_time = (end - start) / trials
                    print(
                        f"T={T},prec.={precision},backend={backend},"
                        f"style={style},time={avg_time:.1E}s,",
                        end="",
                    )
                    act = method(log_in if act_is_log else in_)
                    act = act.tolist()
                    if act_is_log:
                        mM = (
                            (min(x, math.log(y)), max(x, math.log(y)))
                            for (x, y) in zip(act, exp)
                        )
                        diff = [abs(1 - math.exp(m - M)) * math.exp(M) for (m, M) in mM]
                    else:
                        diff = [abs(x - y) for (x, y) in zip(exp, act)]
                    bounds = list(range(0, T + 1, max(1, T // partitions)))
                    bounds[-1] = T
                    for p in range(len(bounds) - 1):
                        diff_p = diff[bounds[p] : bounds[p + 1]]
                        err = sum(diff_p, 0) / (bounds[p + 1] - bounds[p])
                        print(f"partition #{p + 1} error={err:.1E},", end="")
                    print()


if __name__ == "__main__":
    main()
