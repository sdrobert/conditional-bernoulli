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

from typing import Dict, Optional, Type, Tuple

import torch
import param

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
            cond_hist = torch.empty((1, N), dtype=torch.long, device=device)
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


class LstmLmParams(param.Parameterized):

    embedding_size = param.Integer(
        0, bounds=(0, None), doc="The size of token embeddings."
    )
    hidden_size = param.Integer(
        300, bounds=(1, None), doc="The size of the LSTM's hidden states."
    )
    num_layers = param.Integer(3, bounds=(0, None), doc="The number of LSTM layers.")
    merge_method = param.Selector(
        ["mix", "cat"],
        "mix",
        doc="How to combine inputs with embeddings or hidden vectors. mix: softmax for "
        "weights then weighted mixture; cat: concatenate.",
    )


class LstmLm(MixableSequentialLanguageModel):

    input_size: int
    post_input_size: int
    input_name: str
    hidden_name: str
    cell_name: str
    post_input_name: str
    last_hidden_name: str

    def __init__(
        self,
        vocab_size: int,
        input_size: int = 0,
        post_input_size: int = 0,
        params: Optional[LstmLmParams] = None,
        input_name: str = "input",
        hidden_name: str = "hidden",
        cell_name: str = "cell",
        post_input_name: str = "post",
        last_hidden_name: str = "last",
    ):
        if params is None:
            params = LstmLmParams()
        ipe = params.embedding_size
        if params.merge_method == "cat":
            ipe += input_size
        if ipe == 0:
            raise ValueError("Model input dim is 0")
        using_names = [
            input_name,
            post_input_name,
            hidden_name,
            cell_name,
            last_hidden_name,
        ]
        for a, b in itertools.permutations(using_names, 2):
            if not a:
                raise ValueError("No state names can be empty")
            if a == b:
                raise ValueError(f"State name '{a}' matches state name '{b}'.")
        super().__init__(vocab_size)
        self.input_size = input_size
        self.post_input_size = post_input_size
        self.input_name = input_name
        self.hidden_name = hidden_name
        self.cell_name = cell_name
        self.post_input_name = post_input_name
        self.last_hidden_name = last_hidden_name
        if params.embedding_size > 0:
            self.embedder = torch.nn.Embedding(
                vocab_size + 1, params.embedding_size, vocab_size
            )
        else:
            self.add_module("embedder", None)
        if input_size > 0 and params.merge_method == "mix":
            self.input_merger = torch.nn.Linear(input_size, params.embedding_size)
        else:
            self.add_module("input_merger", None)
        self.lstm = torch.nn.LSTM(ipe, params.hidden_size, params.num_layers)
        # We postpone initialization of the cells to the end of init to
        # maintain consistency with reset_parameters.
        in_size = params.hidden_size if params.num_layers else ipe
        if post_input_size > 0:
            if params.merge_method == "mix":
                self.post_merger = torch.nn.Linear(post_input_size, in_size)
            else:
                in_size += post_input_size
                self.add_module("post_merger", None)
        else:
            self.add_module("post_merger", None)
        self.logiter = torch.nn.Linear(in_size, vocab_size)
        in_size = ipe
        cell_list = []
        for idx in range(params.num_layers):
            cell = torch.nn.LSTMCell(in_size, params.hidden_size)
            cell.weight_ih = getattr(self.lstm, f"weight_ih_l{idx}")
            cell.weight_hh = getattr(self.lstm, f"weight_hh_l{idx}")
            cell.bias_ih = getattr(self.lstm, f"bias_ih_l{idx}")
            cell.bias_hh = getattr(self.lstm, f"bias_hh_l{idx}")
            cell_list.append(cell)
            in_size = params.hidden_size
        self.cells = torch.nn.ModuleList(cell_list)

    def reset_parameters(self) -> None:
        if self.embedder is not None:
            self.embedder.reset_parameters()
        if self.input_merger is not None:
            self.input_merger.reset_parameters()
        self.lstm.reset_parameters()
        if self.post_merger is not None:
            self.post_merger.reset_parameters()
        self.logiter.reset_parameters()
        # don't reset cells b/c they share parameters w/ LSTM

    def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
        N, L, H = hist.size(1), len(self.cells), self.lstm.weight_hh_l0.size(1)
        zero = self.lstm.weight_ih_l0.new_zeros(1)
        if self.hidden_name not in prev:
            prev[self.hidden_name] = zero.expand(L, N, H)
        if self.cell_name not in prev:
            prev[self.cell_name] = zero.expand(L, N, H)
        if self.last_hidden_name not in prev:
            prev[self.last_hidden_name] = zero.expand(N, H)
        return prev

    def extract_by_src(self, prev: StateDict, src: torch.Tensor) -> StateDict:
        new_prev = {
            self.hidden_name: prev[self.hidden_name].index_select(1, src),
            self.cell_name: prev[self.cell_name].index_select(1, src),
            self.last_hidden_name: prev[self.last_hidden_name].index_select(0, src),
        }
        if self.input_name in prev and self.input_size:
            x = prev[self.input_name]
            new_prev[self.input_name] = x.index_select(0 if x.dim() == 2 else 1, src)
        if self.post_input_name in prev and self.post_input_size:
            x = prev[self.post_input_name]
            new_prev[self.post_input_name] = x.index_select(
                0 if x.dim() == 2 else 1, src
            )
        return new_prev

    def mix_by_mask(
        self, prev_true: StateDict, prev_false: StateDict, mask: torch.Tensor
    ) -> StateDict:
        mask_2, mask_3 = mask.view(-1, 1), mask.view(1, -1, 1)
        prev = {
            self.hidden_name: mask_3 * prev_true[self.hidden_name]
            + ~mask_3 * prev_false[self.hidden_name],
            self.cell_name: mask_3 * prev_true[self.cell_name]
            + ~mask_3 * prev_false[self.cell_name],
            self.last_hidden_name: mask_2 * prev_true[self.last_hidden_name]
            + ~mask_2 * prev_false[self.last_hidden_name],
        }
        if self.input_name in prev_true and self.input_size:
            x = prev_true[self.input_name]
            assert x.dim() == 3
            prev[self.input_name] = x
        if self.post_input_name in prev_true and self.post_input_size:
            x = prev_true[self.post_input_name]
            prev[self.post_input_name] = x
        return prev

    @staticmethod
    def mix_vectors(a, b):
        w = (a - b).sigmoid()
        return a * w + b * (1 - w)

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, StateDict]:
        T, N = hist.shape
        V = self.vocab_size
        in_ = None
        idx = idx.expand(1, N)
        cur = dict()
        if self.embedder is not None:
            if T == 0:
                tok = hist.new_full((N,), V)  # vocab_size is our sos
            else:
                idx_ = (idx - 1).clamp_min_(0)
                tok = hist.gather(0, idx_).masked_fill_(idx == 0, V)
                tok = tok.squeeze(0)
            in_ = self.embedder(tok)
        if self.input_name in prev and self.input_size:
            x = prev[self.input_name]
            if x.dim() != 2:  # we got all input at once rather than a slice
                cur[self.input_name] = x  # make sure next step gets it too
                x = x.gather(0, idx.unsqueeze(2).expand(1, N, self.input_size)).squeeze(
                    0
                )
            if self.input_merger is not None:
                assert in_ is not None
                x = self.input_merger(x)
                in_ = self.mix_vectors(in_, x)
            else:
                in_ = x if in_ is None else torch.cat([in_, x], 1)
        if in_ is None:
            raise RuntimeError("No input!")
        prev_hidden, prev_cell = prev[self.hidden_name], prev[self.cell_name]
        cur_hidden, cur_cell = [], []
        for lidx in range(len(self.cells)):
            hidden, cell = self.cells[lidx](in_, (prev_hidden[lidx], prev_cell[lidx]))
            cur_hidden.append(hidden)
            cur_cell.append(cell)
            in_ = hidden
        cur[self.hidden_name] = torch.stack(cur_hidden)
        cur[self.cell_name] = torch.stack(cur_cell)
        cur[self.last_hidden_name] = in_
        if self.post_input_name in prev and self.post_input_size:
            x = prev[self.post_input_name]
            if x.dim() != 2:
                cur[self.post_input_name] = x
                x = x.gather(0, idx.unsqueeze(2).expand(1, N, self.input_size)).squeeze(
                    0
                )
            if self.post_merger is not None:
                x = self.post_merger(x)
                in_ = self.mix_vectors(in_, x)
            else:
                in_ = torch.cat([in_, x], 1)
        logits = self.logiter(in_)
        return logits, cur

    def calc_all_hidden(
        self, prev: Dict[str, torch.Tensor], hist: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N = -1
        in_ = None
        if self.embedder is not None:
            if hist is None:
                raise RuntimeError("lm is autoregressive; hist needed")
            N = hist.size(1)
            hist = torch.cat([hist.new_full((1, N), self.vocab_size), hist])
            in_ = self.embedder(hist)
        if self.input_name in prev and self.input_size:
            x = prev[self.input_name]
            if x.dim() != 3:
                raise RuntimeError(f"full input ('{self.input_name}') must be provided")
            if self.input_merger is not None:
                x = self.input_merger(x)
                in_ = self.mix_vectors(in_, x)
            else:
                in_ = x if in_ is None else torch.cat([in_, x], 2)
        if in_ is None:
            raise RuntimeError("No input!")
        return self.lstm(in_, (prev[self.hidden_name], prev[self.cell_name]))[0]

    def calc_logits(self, prev: StateDict, hidden: torch.Tensor) -> torch.Tensor:
        if self.post_input_name in prev and self.post_input_size:
            x = prev[self.post_input_size]
            if x.dim() != 3:
                raise RuntimeError(
                    f"full post input ('{self.post_input_name}') must be provided"
                )
            if self.post_merger is not None:
                x = self.post_merger(x)
                hidden = self.mix_vectors(hidden, x)
            else:
                hidden = x if hidden is None else torch.cat([hidden, x], 2)
        return self.logiter(hidden)

    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.calc_logits(prev, self.calc_all_hidden(prev, hist))


class JointLatentLstmLm(JointLatentLanguageModel):

    latent: LstmLm
    conditional: LstmLm

    def __init__(
        self,
        latent: LstmLm,
        conditional: LstmLm,
        latent_prefix: str = "latent_",
        conditional_prefix: str = "conditional_",
        old_conditional_prefix: str = "old_",
    ):
        if latent.last_hidden_name is None:
            raise ValueError("the latent model must specify last_hidden_name")
        if conditional.input_name != latent_prefix + latent.last_hidden_name:
            raise ValueError(
                "latent hidden state must be input to conditional! Expected "
                f"conditional.input_name to be '{latent_prefix + latent.last_hidden_name}', "
                f"got '{conditional.input_name}'"
            )
        if conditional.input_size != latent.hidden_size:
            raise ValueError(
                "latent hidden state must be input to conditional! Expected "
                f"conditional.input_size to be {latent.hidden_size}, got "
                f"{conditional.input_size}"
            )
        super().__init__(
            latent,
            conditional,
            latent_prefix,
            conditional_prefix,
            old_conditional_prefix,
        )

    def calc_full_log_probs(self, hist: torch.Tensor, prev: StateDict) -> torch.Tensor:
        prev_latent, prev_cond, _, _ = self.split_dicts(prev)
        hidden = self.latent.calc_all_hidden(prev_latent, hist)
        llogits = self.latent.calc_logits(hidden)
        T, N = hist.shape
        device, V = hist.device, self.conditional.vocab_size
        prev_cond[self.conditional.input_name] = hidden
        if T > 0:
            hist = hist.T
            latent_hist = hist != V
            range_ = torch.arange(T, device=device)
            cond_hist = hist[latent_hist]
            cond_idx = latent_hist.long().cumsum(1)
            cond_mask = cond_idx[..., -1:] > range_
            hist = hist.masked_scatter(cond_mask, cond_hist).T
            cond_idx = torch.cat([cond_idx.new_zeros(1, N), cond_idx.T], 0).unsqueeze(2)
        else:
            cond_idx = hist.new_zeros((1,))
        clogits = self.conditional(hist, prev_cond)
        clogits = clogits.gather(0, cond_idx.expand(T + 1, N, V))
        clogits = clogits.log_softmax(2)
        logits = torch.cat([llogits[..., 1:] + clogits, llogits[..., :1]], 2)
        return logits


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


def test_lstm_lm():
    torch.manual_seed(3)

    T, N, V, H = 10, 3, 5, 20
    loss_fn = torch.nn.CrossEntropyLoss()
    for merge_method in ("mix", "cat"):
        for embedding_size, input_size in itertools.permutations(range(0, 10, 5), 2):
            input_posts = [False, True]
            if embedding_size == 0:
                input_posts.pop()
                if merge_method == "mix":
                    continue
            for input_post in input_posts:
                if embedding_size == 0 and input_size == 0:
                    continue
                print(
                    f"merge_method={merge_method}, embedding_size={embedding_size}, "
                    f"input_size={input_size}, input_post={input_post}"
                )
                post_input_size = 0
                input_name = "input"

                if input_post:
                    post_input_size, input_size = input_size, post_input_size
                    input_name = "post"
                params = LstmLmParams(
                    embedding_size=embedding_size,
                    merge_method=merge_method,
                    hidden_size=H,
                )
                hist = torch.randint(V, (T, N))
                prev = {input_name: torch.randn(T, N, V)}
                lm = LstmLm(V, input_size, post_input_size, params)
                logits_exp = lm(hist[:-1], prev)
                loss_exp = loss_fn(logits_exp.flatten(end_dim=-2), hist.flatten())
                g_exp = torch.autograd.grad(loss_exp, lm.parameters())

                logits_act = []
                for t in range(T):
                    logits_act_t, prev = lm(hist[:t], prev, t)
                    logits_act.append(logits_act_t)
                logits_act = torch.stack(logits_act)
                assert logits_act.shape == logits_exp.shape
                assert torch.allclose(logits_exp, logits_act)
                loss_act = loss_fn(logits_act.flatten(end_dim=-2), hist.flatten())
                assert torch.allclose(loss_exp, loss_act)
                g_act = torch.autograd.grad(loss_act, lm.parameters())
                for g_exp_p, g_act_p in zip(g_exp, g_act):
                    assert torch.allclose(g_exp_p, g_act_p)


def test_joint_latent_lstm_lm():
    torch.manual_seed(4)

    T, N, V, H, I, E = 6, 4, 3, 10, 5, 2
    loss_fn = torch.nn.CrossEntropyLoss()
    latent = LstmLm(2, I, 0, H, last_hidden_name="last")
    conditional = LstmLm(V, H, E, input_name="latent_last")
    lm = JointLatentLanguageModel(latent, conditional)
    hist = torch.randint(V + 1, (T, N))
    in_ = torch.rand(T, N, I)
    logits_exp = lm(hist[:-1], {"latent_input": in_})
    assert logits_exp.shape == (T, N, V + 1)
    loss_exp = loss_fn(logits_exp.flatten(end_dim=-2), hist.flatten())
    g_exp = torch.autograd.grad(loss_exp, lm.parameters())

    lm = JointLatentLstmLm(latent, conditional)
    logits_act = lm(hist[:-1], {"latent_input": in_})
    assert logits_act.shape == (T, N, V + 1)
    assert torch.allclose(logits_exp, logits_act, atol=1e-4)
    loss_act = loss_fn(logits_act.flatten(end_dim=-2), hist.flatten())
    g_act = torch.autograd.grad(loss_act, lm.parameters())
    for g_exp_p, g_act_p in zip(g_exp, g_act):
        assert torch.allclose(g_exp_p, g_act_p, atol=1e-4)

