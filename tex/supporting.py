import math
import numpy as np

T, L, p = 10, 2, .5
timit_phone_dur = 0.08
timit_sent_dur = 3.00
frame_length = 0.03

wsj_wpu = 102


def Pt0(t0, T=T, L=L):
    if t0 == 0:
        return 1. if L == 0 else 0.
    return math.comb(t0 - 1, L - 1) * p ** L * (1 - p) ** (t0 - L)


def Pt1(t1, T=T, L=L):
    if t1 == 0:
        return 1. if T == L else 0.
    return math.comb(t1 - 1, T - L - 1) * p ** (L - T + t1) * (1 - p) ** (T - L)


def beta(a, b, x=1):
    if x == 1.:
        return math.factorial(a - 1) * math.factorial(b - 1) / math.factorial(a + b - 1)
    if a == 1:
        return 1. - (1 - x) ** b
    if b == 1:
        return x ** a
    return x ** a * (1 - x) ** (b - 1) / ((b - 1) * beta(a, b - 1)) + beta(a, b - 1, x)


def brute1(t, T=T, L=L):
    return (
        sum(Pt0(t0) * (L - 1) / (t0 - 1) for t0 in range(t + 1, T)) +
        sum(Pt1(t1) for t1 in range(t)) + Pt0(t) * float(t < T) +
        sum(Pt1(t1) * (t1 - T + L) / (t1 - 1) for t1 in range(t + 1, T))
    )


def binom(n=T, k=L, p=p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def Pb1(t, T=T, L=L, p=p):
    assert L < T - L
    s = p
    if t > L:
        s -= p * beta(L, t - L, p)
        if t > T - L:
            s += (1 - p) * beta(T - L, t - T + L, 1 - p)
    return s


def add_probs_to_fig(
        label_secs, frame_secs, total_secs=None,
        label=None, p=None, fig=None, include_expected=False,
        include_L=False):
    from matplotlib import rc
    rc('text', usetex=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(num=fig.number if fig is not None else None)
    if total_secs is not None:
        plt.xlim((0.0, total_secs))
    plt.ylim(bottom=0.0)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\displaystyle P^*(B_t=1)$')
    _, total_secs = plt.xlim()
    expected_label = 'Expected'
    L_label = r'$\displaystyle t=L$'
    if p is None:
        p = 0.5
        if label is None:
            label = f'{frame_secs:.2f}s/frame'
        expected_label += '@' + label
        L_label += '@' + label
        expected_match = True
    elif label is None:
        label = r'$\displaystyle P(B_t=1) =' f'{p:.1f}' '$'
        expected_match = False
    T = int(np.round(total_secs / frame_secs))
    L = int(np.round(total_secs / label_secs))
    t = np.linspace(frame_secs, total_secs, T)
    Pb = np.array([Pb1(t, T=T, L=L, p=p) for t in range(1, T + 1)])
    s = plt.scatter(t, Pb, label=f'Actual@{label}', marker='.')
    if include_expected:
        plt.hlines(
            L / T, 0.0, total_secs, label=expected_label,
            linestyle='--', color=s.get_ec() if expected_match else 'black')
    if include_L:
        plt.vlines(
            t[L - 1], 0.0, 1.0, label=L_label,
            linestyle='--', color=s.get_ec() if expected_match else 'black')
    plt.legend()
    return fig


def compare_probs(label_secs, frame_secs, p, *rest, total_secs=3.):
    import matplotlib.pyplot as plt
    fig = None
    ps = (p,) + rest
    for i, p in enumerate(ps):
        last = i == len(ps) - 1
        fig = add_probs_to_fig(
            label_secs, frame_secs, total_secs,
            p=p, fig=fig, include_expected=last, include_L=last
        )
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=(len(ps) + 3) // 2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    return fig


def compare_shifts(label_secs, frame_secs, *rest, total_secs=3.):
    import matplotlib.pyplot as plt
    fig = None
    frame_secss = (frame_secs,) + rest
    for frame_secs in frame_secss:
        fig = add_probs_to_fig(
            label_secs, frame_secs, total_secs=total_secs,
            fig=fig, include_expected=True, include_L=True
        )
    plt.ylim((0., 0.6))
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=len(frame_secss), mode="expand", borderaxespad=0.)
    plt.tight_layout()
    return fig
