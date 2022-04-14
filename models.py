# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models and model wrappers"""

from typing import Dict, Type, Tuple
import torch

import pydrobert.torch.config as config
from pydrobert.torch.modules import SequentialLanguageModel


def build_suffix_forcing(
    cls: Type[SequentialLanguageModel],
    total_count_name: str = "total",
    given_count_name: str = "given",
) -> Type[SequentialLanguageModel]:
    if not issubclass(cls, SequentialLanguageModel):
        raise ValueError(f"Class {cls.__name__} is not a SequentialLanguageModel")

    class _SuffixForcing(cls):
        f"""Fixed cardinality version of {cls.__name__}"""

        __constants__ = ["_tn", "_gn"]
        if hasattr(cls, "__constants__"):
            __constants__ += cls.__constants__

        _tn = total_count_name
        _gn = given_count_name

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.vocab_size != 2:
                raise ValueError(f"vocab_size must be 2, got {self.vocab_size}")

        if hasattr(cls, "extract_by_source"):

            def extract_by_source(
                self, prev: Dict[str, torch.Tensor], src: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
                total, given = prev[self._tn], prev[self._gn]
                new_prev = super().extract_by_source(prev, src)
                total, given = total.index_select(0, src), given.index_select(0, src)
                new_prev[self._tn], new_prev[self._gn] = total, given
                return new_prev

        if hasattr(cls, "mix_by_mask"):

            def mix_by_mask(
                self,
                prev_true: Dict[str, torch.Tensor],
                prev_false: Dict[str, torch.Tensor],
                mask: torch.Tensor,
            ) -> Dict[str, torch.Tensor]:
                total, given = prev_true[self._tn], prev_true[self._gn]
                assert (total == prev_false[self._tn]).all()
                assert (given == prev_false[self._gn]).all()
                new_prev = super().mix_by_mask(prev_true, prev_false)
                new_prev[self._tn], new_prev[self._gn] = total, given
                return new_prev

        def calc_idx_log_probs(
            self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            total, given = prev[self._tn], prev[self._gn]
            logits, cur = super().calc_idx_log_probs(hist, prev, idx)
            N = hist.size(1)
            device = hist.device
            hist = torch.cat(
                [torch.zeros((1, N), device=device, dtype=torch.long), hist]
            )
            count = hist.cumsum(0).gather(0, idx.expand(1, N)).squeeze(0)
            remaining_count = given - count
            remaining_space = total - idx
            force_low = (remaining_count <= 0).unsqueeze(1)
            force_high = (remaining_count == remaining_space).unsqueeze(1) & ~force_low
            high_config = torch.tensor([config.EPS_NINF, config.EPS_0], device=device)
            low_config = torch.tensor([config.EPS_0, config.EPS_NINF], device=device)
            logits = (
                (~force_low & ~force_high) * logits
                + force_low * low_config
                + force_high * high_config
            )
            cur[self._tn] = total
            cur[self._gn] = given
            return logits, cur

        def calc_full_log_probs(
            self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            total, given = prev[self._tn], prev[self._gn]
            logits = super().calc_full_log_probs(hist, prev)
            T, N = hist.shape
            device = hist.device
            # prepend 0 to hist (for empty prefix count)
            hist = torch.cat(
                [torch.zeros((1, N), device=device, dtype=torch.long), hist]
            )
            # set values of hist beyond total to zero (should be padding)
            idx = torch.arange(-1, T, device=device)
            hist.masked_fill_(idx.unsqueeze(1) >= total, 0)
            count = hist.cumsum(0)
            remaining_count = given.unsqueeze(0) - count
            remaining_space = total.unsqueeze(0) - (idx + 1).unsqueeze(1)
            force_low = (remaining_count <= 0).unsqueeze(2)
            force_high = (remaining_count == remaining_space).unsqueeze(2) & ~force_low
            high_config = torch.tensor([config.EPS_NINF, config.EPS_0], device=device)
            low_config = torch.tensor([config.EPS_0, config.EPS_NINF], device=device)
            logits = (
                (~force_low & ~force_high) * logits
                + force_low * low_config
                + force_high * high_config
            )
            return logits

    return _SuffixForcing


## TESTS


def test_build_suffix_forcing():
    from itertools import product
    from math import factorial
    from pydrobert.torch.modules import RandomWalk
    from pydrobert.torch.distributions import SequentialLanguageModelDistribution

    class UniformBinary(SequentialLanguageModel):
        def __init__(self):
            super().__init__(2)

        def calc_idx_log_probs(
            self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            logits = torch.ones((hist.size(1), 2), device=hist.device)
            return logits, prev

    def beta(a, b, x=1):
        if a == 0 or b == 0:
            return 0
        if x == 1.0:
            return factorial(a - 1) * factorial(b - 1) / factorial(a + b - 1)
        if a == 1:
            return 1.0 - (1 - x) ** b
        if b == 1:
            return x ** a
        return x ** a * (1 - x) ** (b - 1) / ((b - 1) * beta(a, b - 1)) + beta(
            a, b - 1, x
        )

    # For independent probability p, I proved that
    #
    #   P(b_t=1|T, L, p) = p + I[t > L] p B(p; L, t - L)
    #                        + I[t > T - L] (1 - p) B(1 - p; T - L; t - T + L)
    #
    # In commit f25329. B is the regularized incomplete beta function
    # https://en.wikipedia.org/wiki/Beta_function
    # Look for pdf in tex/ folder
    def Pb1(t, T, L, p):
        if t > T or L > T or L == 0:
            return 0  # this is our
        s = p
        if t > L:
            s -= p * beta(L, t - L, p)
        if t > T - L:
            s += (1 - p) * beta(T - L, t - T + L, 1 - p)
        return s

    torch.manual_seed(1)
    T, N, M = 5, 2, 2 ** 13
    zero = torch.zeros(1)
    uniform_binary = UniformBinary()
    walk = RandomWalk(uniform_binary, max_iters=T)
    dist = SequentialLanguageModelDistribution(walk, N)
    support = dist.enumerate_support()
    log_probs = dist.log_prob(support)
    assert log_probs.shape == (2 ** T, N)
    assert torch.allclose(log_probs.logsumexp(0), zero)
    sample = dist.sample([M])
    act = sample.float().mean(0)
    assert torch.allclose(act, zero + 0.5, atol=1e-2)

    total = torch.randint(T + 1, (N,))
    given = (torch.rand(N) * (total + 1)).long()
    assert (given <= total).all()
    suffix_forcing = build_suffix_forcing(UniformBinary)()
    walk = RandomWalk(suffix_forcing, max_iters=T)
    dist = SequentialLanguageModelDistribution(
        walk, N, {"total": total, "given": given}
    )
    log_probs = dist.log_prob(support)
    assert log_probs.shape == (2 ** T, N)
    assert torch.allclose(log_probs.logsumexp(0), zero)
    sample = dist.sample([M])
    assert sample.shape == (M, N, T)
    counts = sample.sum(2)
    assert (counts == given.unsqueeze(0)).all()
    idx = torch.arange(T)
    counts = (sample * (total.unsqueeze(1) > idx)).sum(2)
    assert (counts == given.unsqueeze(0)).all()
    exp = torch.tensor(
        [
            Pb1(t + 1, T_, L, 0.5)
            for (T_, L, t) in product(range(T + 1), range(T + 1), range(T))
        ]
    ).view((T + 1) ** 2, T)
    exp = exp[total * (T + 1) + given]
    assert exp.shape == (N, T)
    act = sample.float().mean(0)
    assert torch.allclose(act, exp, atol=1e-2)

