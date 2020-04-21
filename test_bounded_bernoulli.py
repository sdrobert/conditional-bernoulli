import torch
import conditional_bernoulli
import bounded_bernoulli


def to_one_hot(b_index, len_, dim=0):
    shape = list(b_index.shape)
    shape[dim] = len_ + 1
    b_onehot = torch.zeros(*shape, device=b_index.device, dtype=torch.float)
    b_onehot.scatter_(dim, (b_index + 1).clamp(0), 1.)
    _, b_onehot = b_onehot.split([1, len_])
    return b_onehot


def test_sample():
    torch.manual_seed(40272)
    T, N = 10, 100000
    w = torch.rand(T)
    w[::2] = 0.
    counts = torch.tensor([4, 5]).unsqueeze(0).expand(N, 2)

    s = to_one_hot(bounded_bernoulli.sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    ), T)
    assert torch.all(s.int().sum(0) == counts)
    props1 = s.mean(1)

    s = conditional_bernoulli.sample(
        w.view(T, 1, 1).expand(T, N, 2), counts
    )
    assert torch.all(s.int().sum(0) == counts)
    props2 = s.mean(1)

    assert torch.allclose(props1, props2, atol=1e-2)
