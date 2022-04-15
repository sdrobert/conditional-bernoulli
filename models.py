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

import itertools

from typing import Dict, Type, Tuple

import torch

import pydrobert.torch.config as config
from pydrobert.torch.modules import (
    SequentialLanguageModel,
    ExtractableSequentialLanguageModel,
    MixableSequentialLanguageModel,
)

StateDict = Dict[str, torch.Tensor]


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
                self, prev_true: StateDict, prev_false: StateDict, mask: torch.Tensor,
            ) -> StateDict:
                total, given = prev_true[self._tn], prev_true[self._gn]
                assert (total == prev_false[self._tn]).all()
                assert (given == prev_false[self._gn]).all()
                new_prev = super().mix_by_mask(prev_true, prev_false)
                new_prev[self._tn], new_prev[self._gn] = total, given
                return new_prev

        def calc_idx_log_probs(
            self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
        ) -> Tuple[torch.Tensor, StateDict]:
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
            self, hist: torch.Tensor, prev: StateDict
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


class JointLatentLanguageModel(ExtractableSequentialLanguageModel):

    __constants__ = ["latent_prefix", "conditional_prefix", "old_conditional_prefix"]
    latent_prefix: str
    conditional_prefix: str
    old_conditional_prefix: str

    def __init__(
        self,
        latent: ExtractableSequentialLanguageModel,
        conditional: MixableSequentialLanguageModel,
        latent_prefix: str = "latent_",
        conditional_prefix: str = "conditional_",
        old_conditional_prefix: str = "old_",
    ):
        if latent.vocab_size != 2:
            raise ValueError(
                f"Expected latent to have vocab size 2, got {latent.vocab_size}"
            )
        for a, b in itertools.permutations(
            (latent_prefix, conditional_prefix, old_conditional_prefix), 2
        ):
            if a.startswith(b):
                raise ValueError(
                    f"Prefix '{a}' starts with prefix '{b}'. Cannot unambiguously "
                    "extract states."
                )
        super().__init__(conditional.vocab_size + 1)
        self.latent = latent
        self.conditional = conditional
        self.latent_prefix = latent_prefix
        self.conditional_prefix = conditional_prefix
        self.old_conditional_prefix = old_conditional_prefix

    def reset_parameters(self) -> None:
        if hasattr(self.latent, "reset_parameters"):
            self.latent.reset_parameters()
        if hasattr(self.conditional, "reset_parameters"):
            self.conditional.reset_parameters()

    def split_dicts(
        self, prev: StateDict
    ) -> Tuple[StateDict, StateDict, StateDict, StateDict]:
        prev_latent, prev_cond, prev_old, prev_rest = dict(), dict(), dict(), dict()
        for key, value in prev.items():
            if key.startswith(self.latent_prefix):
                prev_latent[key[len(self.latent_prefix) :]] = value
            elif key.startswith(self.conditional_prefix):
                prev_cond[key[len(self.conditional_prefix) :]] = value
            elif key.startswith(self.old_conditional_prefix):
                prev_old[key[len(self.old_conditional_prefix) :]] = value
            else:
                prev_rest[key] = value
        return prev_latent, prev_cond, prev_old, prev_rest

    def merge_dicts(
        self,
        prev_latent: StateDict,
        prev_cond: StateDict,
        prev_old: StateDict,
        prev_rest: StateDict,
    ) -> StateDict:
        prev = prev_rest.copy()
        prev.update((self.latent_prefix + k, v) for (k, v) in prev_latent.items())
        prev.update((self.conditional_prefix + k, v) for (k, v) in prev_cond.items())
        prev.update((self.old_conditional_prefix + k, v) for (k, v) in prev_old.items())
        return prev

    def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
        prev_latent, prev_cond, prev_old, prev_rest = self.split_dicts(prev)
        prev_latent = self.latent.update_input(prev_latent, hist)
        prev_cond = self.conditional.update_input(prev_cond, hist)
        return self.merge_dicts(prev_latent, prev_cond, prev_old, prev_rest)

    def extract_by_src(self, prev: StateDict, src: torch.Tensor) -> StateDict:
        prev_latent, prev_cond, prev_old, prev_rest = self.split_dicts(prev)
        prev_latent = self.latent.extract_by_src(prev_latent, src)
        prev_cond = self.conditional.extract_by_src(prev_cond, src)
        prev_old = self.conditional.extract_by_src(prev_old, src)
        return self.merge_dicts(prev_latent, prev_cond, prev_old, prev_rest)

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, StateDict]:
        prev_latent, prev_cond, prev_old, prev_rest = self.split_dicts(prev)
        T, N = hist.shape
        device = hist.device
        latent_hist = hist != self.conditional.vocab_size
        idx = idx.expand(N)
        if (idx == 0).all():
            cond_idx = idx
            cond_hist = torch.empty((T, N), dtype=torch.long, device=device)
        else:
            range_ = torch.arange(T, device=device)
            cond_hist = hist.T[latent_hist.T & (idx.unsqueeze(1) > range_)]
            idx_ = (idx - 1).clamp_min(0).unsqueeze(0)
            gt_0 = idx > 0
            is_new = latent_hist.gather(0, idx_).squeeze(0) & gt_0
            latent_hist = latent_hist.long()
            cond_idx = latent_hist.cumsum(0).gather(0, idx_).squeeze(0) * gt_0
            cond_mask = cond_idx.unsqueeze(1) > range_
            cond_hist = hist.T.masked_scatter(cond_mask, cond_hist).T
            prev_cond = self.conditional.mix_by_mask(prev_cond, prev_old, is_new)
        llogits, cur_latent = self.latent.calc_idx_log_probs(
            latent_hist.long(), prev_latent, idx
        )
        prev_enhanced = prev_cond.copy()
        prev_enhanced.update(
            (self.latent_prefix + k, v) for (k, v) in cur_latent.items()
        )
        clogits, cur_cond = self.conditional.calc_idx_log_probs(
            cond_hist, prev_enhanced, cond_idx
        )
        # don't have to normalize llogits if logits is normalized later
        clogits = clogits.log_softmax(1)
        logits = torch.cat([llogits[:, 1:] + clogits, llogits[:, :1]], 1)
        cur = self.merge_dicts(cur_latent, cur_cond, prev_cond, prev_rest)
        return logits, cur


## TESTS


def test_build_suffix_forcing():
    from scipy.special import betainc
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

    # For independent probability p, I proved that
    #
    #   P(b_t=1|T, L, p) = p + I[t > L] p B(p; L, t - L)
    #                        + I[t > T - L] (1 - p) B(1 - p; T - L; t - T + L)
    #
    # In commit f25329. B is the regularized incomplete beta function
    # https://en.wikipedia.org/wiki/Beta_function
    # Look for pdf in tex/ folder
    def Pb1(t, T, L, p):
        if t > T:
            # technically undefined, but we allow 0-padding past T
            return 0
        if L > T:
            return float("nan")
        s = p
        if t > L:
            s -= p * (betainc(L, t - L, p) if L > 0 else 1)
        if t > T - L:
            s += (1 - p) * (betainc(T - L, t - T + L, 1 - p) if T - L > 0 else 1)
        return s

    torch.manual_seed(1)
    T, N, M = 5, 10, 2 ** 13
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
    assert torch.allclose(act, zero + 0.5, atol=2e-2)

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
    exp = (
        torch.tensor(
            [
                Pb1(t + 1, T_, L, 0.5)
                for (T_, L, t) in itertools.product(
                    range(T + 1), range(T + 1), range(T)
                )
            ]
        )
        .view((T + 1) ** 2, T)
        .float()
    )
    exp = exp[total * (T + 1) + given]
    assert exp.shape == (N, T)
    act = sample.float().mean(0)
    assert torch.allclose(act, exp, atol=2e-2), (act - exp).max()


def test_joint_latent_model():
    from pydrobert.torch.modules import RandomWalk
    from pydrobert.torch.distributions import SequentialLanguageModelDistribution

    torch.manual_seed(2)

    class DummyLatent(ExtractableSequentialLanguageModel):

        embedding_size: int

        def __init__(self, embedding_size: int = 128):
            super().__init__(2)
            self.embedding_size = embedding_size
            self.embedder = torch.nn.Embedding(2, embedding_size)
            self.ff = torch.nn.Linear(2 * self.embedding_size, 2)

        def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
            if len(prev):
                return prev
            N, device = hist.size(1), hist.device
            return {"embedding": torch.zeros((N, self.embedding_size), device=device)}

        def extract_by_src(self, prev: StateDict, src: torch.Tensor) -> StateDict:
            return {"embedding": prev["embedding"].index_select(src)}

        def calc_idx_log_probs(
            self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
        ) -> Tuple[torch.Tensor, StateDict]:
            T, N = hist.shape
            if T == 0:
                cur_embedding = torch.zeros_like(prev["embedding"])
            else:
                nonzero = idx != 0
                tok = hist.gather(0, (idx - 1).clamp_min_(0).expand(1, N)).squeeze(0)
                cur_embedding = self.embedder(tok * nonzero)
                cur_embedding = cur_embedding * nonzero.unsqueeze(1)
            logits = self.ff(torch.cat([cur_embedding, prev["embedding"]], 1))
            return logits, {"embedding": cur_embedding}

    class DummyConditional(MixableSequentialLanguageModel):

        embedding_size: int
        hidden_size: int

        def __init__(
            self, vocab_size: int, embedding_size: int = 128, hidden_size: int = 128
        ):
            super().__init__(vocab_size)
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.embedder = torch.nn.Embedding(vocab_size, embedding_size)
            self.rnn = torch.nn.RNNCell(2 * embedding_size, hidden_size)
            self.ff = torch.nn.Linear(hidden_size, vocab_size)

        def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
            if len(prev):
                return prev
            N, device = hist.size(1), hist.device
            return {"hidden": torch.zeros((N, self.hidden_size), device=device)}

        def extract_by_src(self, prev: StateDict, src: torch.Tensor) -> StateDict:
            return {"hidden": prev["hidden"].index_select(0, src)}

        def mix_by_mask(
            self, prev_true: StateDict, prev_false: StateDict, mask: torch.Tensor
        ) -> StateDict:
            mask = mask.unsqueeze(1)
            return {"hidden": mask * prev_true["hidden"] + ~mask * prev_false["hidden"]}

        def calc_idx_log_probs(
            self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
        ) -> Tuple[torch.Tensor, StateDict]:
            T, N = hist.shape
            if T == 0:
                cur_embedding = torch.zeros_like(prev["latent_embedding"])
            else:
                nonzero = idx != 0
                tok = hist.gather(0, (idx - 1).clamp_min_(0).expand(1, N)).squeeze(0)
                cur_embedding = self.embedder(tok * nonzero)
                cur_embedding = cur_embedding * nonzero.unsqueeze(1)
            input_ = torch.cat([cur_embedding, prev["latent_embedding"]], 1)
            hidden = self.rnn(input_, prev["hidden"])
            logits = self.ff(hidden)
            return logits, {"hidden": hidden}

    T, V = 5, 5
    latent = DummyLatent()
    conditional = DummyConditional(V)
    joint = JointLatentLanguageModel(latent, conditional)
    walk = RandomWalk(joint, max_iters=T)
    dist = SequentialLanguageModelDistribution(walk)
    support = dist.enumerate_support()
    assert support.shape == ((V + 1) ** T, T)
    log_probs = dist.log_prob(support)
    assert torch.allclose(log_probs.logsumexp(0), torch.tensor(0.0))

