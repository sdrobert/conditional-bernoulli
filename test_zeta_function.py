import zeta_function
import pytest
import torch


@pytest.fixture(
    scope="module",
    params=[
        "full-f",
        pytest.param("partial-f", marks=pytest.mark.xfail),
        pytest.param("tensor-f", marks=pytest.mark.xfail),
    ],
)
def zeta(request):
    if request.param == "full-f":

        def _zeta(w, f, L=None, K=None, full=False) -> torch.Tensor:
            L, N, T = f.shape[:3]
            K = len(f.shape) - 2
            f = (
                f.reshape(L, N, T * K)
                .transpose(1, 2)
                .reshape(*((L,) + (T,) * K + (N,)))
            )

            def _f(ell, *slice_):
                Kp = min(ell, K)
                assert len(slice_) == Kp
                x = f[ell - 1]  # (T,) * K + (N,)
                # recall indices are reversed in the tensor
                for _ in range(K - Kp):
                    x = x[0]  # all indices (K', K] are just copies of one another,
                    # so just pick something
                slice_ = list(slice_)
                while len(slice_):
                    x = x[slice_.pop(-1) - 1]
                assert x.dim() == 1  # (N,)
                return x

            return zeta_function.zeta(w, _f, L, K, full)

        return _zeta
    elif request.param == "partial-f":

        def _zeta(w, f, L=None, K=None, full=False) -> torch.Tensor:
            L = f.shape[0]

            def _f(ell):
                return f[ell - 1]

            return zeta_function.zeta(w, _f, L, None, full)

        return _zeta
    else:
        return zeta_function.zeta


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_zeta_values_k1(zeta, device, batch_size):
    w = torch.rand((batch_size, 4), device=device)
    f = torch.rand((6, batch_size, 4), device=device)
    w.requires_grad_(True)
    f.requires_grad_(True)
    w1, w2, w3, w4 = torch.unbind(w, 1)
    f1, f2, f3, f4, _, _ = torch.unbind(f, 0)
    f11, f12, f13, f14 = torch.unbind(f1, 1)
    _, f22, f23, f24 = torch.unbind(f2, 1)
    _, _, f33, f34 = torch.unbind(f3, 1)
    f44 = f4[:, -1]
    ones = torch.ones_like(w1)
    zeros = torch.zeros_like(w1)
    C_exp = torch.stack(
        [
            ones,
            f11 * w1 + f12 * w2 + f13 * w3 + f14 * w4,
            f11 * w1 * (f22 * w2 + f23 * w3 + f24 * w4)
            + f12 * w2 * (f23 * w3 + f24 * w4)
            + f13 * w3 * f24 * w4,
            f11 * w1 * (f22 * w2 * (f33 * w3 + f34 * w4) + f23 * w3 * f34 * w4)
            + f12 * w2 * f23 * w3 * f34 * w4,
            f11 * w1 * f22 * w2 * f33 * w3 * f44 * w4,
            zeros,
            zeros,
        ],
        0,
    )
    (gw_exp, gf_exp) = torch.autograd.grad(C_exp, [w, f], torch.ones_like(C_exp))
    w.requires_grad_(True)
    f.requires_grad_(True)
    C_act = zeta(w, f, 6, 1)
    assert C_exp.shape == C_act.shape
    assert torch.allclose(C_exp, C_act, atol=1e-5)
    (gw_act, gf_act) = torch.autograd.grad(C_act, [w, f], torch.ones_like(C_act))
    assert torch.allclose(gw_exp, gw_act, atol=1e-5)
    assert torch.allclose(gf_exp, gf_act, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 30])
def test_zeta_history_k1(zeta, device, batch_size):
    w = torch.rand((batch_size, 4), device=device)
    w1, w2, w3, w4 = w.unbind(1)
    f = torch.rand((2, batch_size, 4), device=device)
    f1, f2 = torch.unbind(f, 0)
    f11, f12, f13, f14 = torch.unbind(f1, 1)
    _, f22, f23, f24 = torch.unbind(f2, 1)
    ones = torch.ones((batch_size,), device=device)
    zeros = torch.zeros_like(ones)
    Z_exp = torch.stack(
        [
            torch.stack([ones, ones, ones, ones, ones], 1),
            torch.stack(
                [
                    zeros,
                    f11 * w1,
                    f11 * w1 + f12 * w2,
                    f11 * w1 + f12 * w2 + f13 * w3,
                    f11 * w1 + f12 * w2 + f13 * w3 + f14 * w4,
                ],
                1,
            ),
            torch.stack(
                [
                    zeros,
                    zeros,
                    f11 * w1 * f22 * w2,
                    f11 * w1 * f22 * w2 + f11 * w1 * f23 * w3 + f12 * w2 * f23 * w3,
                    f11 * w1 * f22 * w2
                    + f11 * w1 * f23 * w3
                    + f11 * w1 * f24 * w4
                    + f12 * w2 * f23 * w3
                    + f12 * w2 * f24 * w4
                    + f13 * w3 * f24 * w4,
                ],
                1,
            ),
        ],
        0,
    )
    Z_act = zeta(w, f, 2, 1, full=True)
    assert torch.allclose(Z_exp, Z_act, atol=1e-5)
