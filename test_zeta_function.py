import zeta_function
import pytest
import torch
import itertools


@pytest.fixture(
    scope="module", params=["full-f", "tensor-f"],
)
def zeta(request):
    if request.param == "full-f":

        def _zeta(w, f, L=None, K=None, full=False) -> torch.Tensor:
            L, N, T = f.shape[:3]
            K = len(f.shape) - 2
            f = (
                f.reshape(L, N, T ** K)
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
    Z_exp = torch.stack(
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
    (gw_exp, gf_exp) = torch.autograd.grad(Z_exp, [w, f], torch.ones_like(Z_exp))
    w.requires_grad_(True)
    f.requires_grad_(True)
    Z_act = zeta(w, f, 6, 1)
    assert Z_exp.shape == Z_act.shape
    assert torch.allclose(Z_exp, Z_act, atol=1e-5)
    (gw_act, gf_act) = torch.autograd.grad(Z_act, [w, f], torch.ones_like(Z_act))
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


@pytest.mark.parametrize("batch_size", [1, 2, 30])
def test_zeta_values_k3(zeta, device, batch_size):
    w = torch.rand((batch_size, 5), device=device)
    w.requires_grad_(True)
    w1, w2, w3, w4, w5 = torch.unbind(w, 1)
    # we choose f such that the only nonzero values occur when t_ell <= t_{ell - 1} + 1
    # t_ell <= t_{ell - 1} are invalid event values, so zeta should filter 'em
    # recall f is indexed in reverse order.
    f_1 = torch.ones((1, 5, 5, 5), device=device)
    f_2 = torch.tensor(
        [
            float(u_1 + 1 >= u_2)
            for _, u_2, u_1 in itertools.product(range(5), repeat=3)
        ],
        device=device,
    ).view(1, 5, 5, 5)
    f_ge_3 = (
        torch.tensor(
            [
                float(u_1 + 2 >= u_2 + 1 >= u_3)
                for u_3, u_2, u_1 in itertools.product(range(5), repeat=3)
            ],
            device=device,
        )
        .view(1, 5, 5, 5)
        .expand(4, 5, 5, 5)
    )
    ones = torch.ones_like(w1)
    zeros = torch.zeros_like(w1)
    f = torch.cat([f_1, f_2, f_ge_3], 0).unsqueeze(1).expand(6, batch_size, 5, 5, 5)
    # N.B. ell == 3 does not include [w1, w3, w4, w5] b/c that would require
    # [w1, w3, w4] at ell == 2 to be nonzero
    Z_exp = torch.stack(
        [
            ones,
            w1 + w2 + w3 + w4 + w5,
            w1 * w2 + w2 * w3 + w3 * w4 + w4 * w5,
            w1 * w2 * w3 + w2 * w3 * w4 + w3 * w4 * w5,
            w1 * w2 * w3 * w4 + w2 * w3 * w4 * w5,
            w1 * w2 * w3 * w4 * w5,
            zeros,
        ],
        0,
    )
    (g_exp,) = torch.autograd.grad(Z_exp, w, torch.ones_like(Z_exp))
    w.requires_grad_(True)
    Z_act = zeta(w, f)
    assert Z_exp.shape == Z_act.shape
    assert torch.allclose(Z_exp, Z_act, atol=1e-5)
    (g_act,) = torch.autograd.grad(Z_act, w, torch.ones_like(Z_act))
    assert torch.allclose(g_exp, g_act, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 30])
def test_zeta_history_k2(zeta, device, batch_size):
    w = torch.rand((batch_size, 4), device=device)
    w1, w2, w3, w4 = w.unbind(1)
    f1 = torch.rand((batch_size, 4), device=device)
    f2 = torch.rand((batch_size, 4, 4), device=device)
    f = torch.stack([f1.unsqueeze(1).expand(batch_size, 4, 4), f2], 0)
    f11, f12, f13, f14 = torch.unbind(f1, 1)
    _, f22, f23, f24 = torch.unbind(f2, 1)
    f212 = f22[:, 0]
    f213, f223, _, _ = torch.unbind(f23, 1)
    f214, f224, f234, _ = torch.unbind(f24, 1)
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
                    f11 * w1 * f212 * w2,
                    f11 * w1 * f212 * w2 + f11 * w1 * f213 * w3 + f12 * w2 * f223 * w3,
                    f11 * w1 * f212 * w2
                    + f11 * w1 * f213 * w3
                    + f11 * w1 * f214 * w4
                    + f12 * w2 * f223 * w3
                    + f12 * w2 * f224 * w4
                    + f13 * w3 * f234 * w4,
                ],
                1,
            ),
        ],
        0,
    )
    Z_act = zeta(w, f, full=True)
    assert Z_exp.shape == Z_act.shape
    assert torch.allclose(Z_exp, Z_act, atol=1e-5)
