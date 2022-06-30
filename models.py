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

from typing import Dict, Optional, Tuple

import torch
import param

import pydrobert.torch.config as config
from pydrobert.torch.modules import (
    SequentialLanguageModel,
    ExtractableSequentialLanguageModel,
    MixableSequentialLanguageModel,
)

import distributions

StateDict = Dict[str, torch.Tensor]


class SuffixForcingWrapper(SequentialLanguageModel):

    lm: SequentialLanguageModel
    total_count_name: str
    given_count_name: str

    def __init__(
        self,
        lm: SequentialLanguageModel,
        total_count_name: str = "total",
        given_count_name: str = "given",
    ):
        super().__init__(lm.vocab_size)
        self.lm = lm
        self.total_count_name = total_count_name
        self.given_count_name = given_count_name

    def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
        total, given = prev[self.total_count_name], prev[self.given_count_name]
        prev = self.lm.update_input(prev, hist)
        prev[self.total_count_name], prev[self.given_count_name] = total, given
        return prev

    def extract_by_src(self, prev: StateDict, src: torch.Tensor) -> StateDict:
        prev = self.lm.extract_by_src(prev, src)
        total, given = prev[self.total_count_name], prev[self.given_count_name]
        total, given = total.index_select(0, src), given.index_select(0, src)
        prev[self.total_count_name], prev[self.given_count_name] = total, given
        return prev

    def mix_by_mask(
        self, prev_true: StateDict, prev_false: StateDict, mask: torch.Tensor,
    ) -> StateDict:
        prev = super().mix_by_mask(prev_true, prev_false)
        total = prev_true[self.total_count_name]
        given = prev_true[self.given_count_name]
        # assert (total == prev_false[self.total_count_name]).all()
        # assert (given == prev_false[self.given_count_name]).all()
        prev[self.total_count_name], prev[self.given_count_name] = total, given
        return prev

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, StateDict]:
        total, given = prev[self.total_count_name], prev[self.given_count_name]
        logits, cur = self.lm.calc_idx_log_probs(hist, prev, idx)
        N = hist.size(1)
        device = hist.device
        hist = torch.cat([torch.zeros((1, N), device=device, dtype=torch.long), hist])
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
        cur[self.total_count_name] = total
        cur[self.given_count_name] = given
        return logits, cur

    def calc_full_log_probs(self, hist: torch.Tensor, prev: StateDict) -> torch.Tensor:
        total, given = prev[self.total_count_name], prev[self.given_count_name]
        logits = self.lm.calc_full_log_probs(hist, prev)
        T, N = hist.shape
        device = hist.device
        # prepend 0 to hist (for empty prefix count)
        hist = torch.cat([torch.zeros((1, N), device=device, dtype=torch.long), hist])
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


def extended_hist_to_conditional(
    hist: torch.Tensor,
    hist_lens: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
    batch_first: bool = False,
    latent_hist: Optional[torch.Tensor] = None,
    pad_value: int = config.INDEX_PAD_VALUE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latent_hist is None:
        assert vocab_size is not None
        latent_hist = hist != vocab_size
    if not batch_first:
        hist, latent_hist = hist.T, latent_hist.T
    T = hist.size(1)
    range_ = torch.arange(T, device=hist.device)
    if hist_lens is not None:
        latent_hist = latent_hist & (hist_lens.unsqueeze(1) > range_)
    cond_hist_ = hist[latent_hist]
    cond_lens = latent_hist.long().sum(1)
    cond_mask = cond_lens.unsqueeze(1) > range_
    cond_hist = torch.full_like(hist, pad_value).masked_scatter(cond_mask, cond_hist_)
    if not batch_first:
        cond_hist = cond_hist.T
    return cond_hist, cond_lens


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
        if not prev_old:
            prev_old = prev_cond.copy()
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
        if T == 0:
            cond_idx = idx
            cond_hist = torch.empty((0, N), dtype=torch.long, device=device)
        else:
            cond_hist, cond_lens = extended_hist_to_conditional(
                hist, idx, latent_hist=latent_hist
            )
            idx_ = (idx - 1).clamp_min(0).unsqueeze(0)
            gt_0 = idx > 0
            is_new = latent_hist.gather(0, idx_).squeeze(0) & gt_0
            cond_idx = cond_lens * gt_0
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
        clogits = clogits.log_softmax(1)
        logits = torch.cat([llogits[:, 1:] + clogits, llogits[:, :1]], 1)
        cur = self.merge_dicts(cur_latent, cur_cond, prev_cond, prev_rest)
        return logits, cur

    def calc_log_likelihood_given_latents(
        self,
        latent_hist: torch.Tensor,
        cond_hist: torch.Tensor,
        prev: StateDict = dict(),
    ) -> torch.Tensor:
        T, N = latent_hist.shape
        L, N_ = cond_hist.shape
        V = self.conditional.vocab_size
        assert N == N_
        device = latent_hist.device
        latent_hist = latent_hist.long()
        prev = self.update_input(prev, latent_hist)
        prev_latent, prev_cond, _, _ = self.split_dicts(prev)
        cond_lens = latent_hist.sum(0)
        ll = torch.zeros(N, device=device)
        cond_idx = torch.zeros(N, device=device, dtype=torch.long)
        for t in torch.arange(T, device=device):
            is_new = latent_hist[t].bool()
            _, prev_latent = self.latent.calc_idx_log_probs(
                latent_hist.long(), prev_latent, t
            )
            prev_enhanced = prev_cond.copy()
            prev_enhanced.update(
                (self.latent_prefix + k, v) for (k, v) in prev_latent.items()
            )
            clogits, cur_cond = self.conditional.calc_idx_log_probs(
                cond_hist, prev_enhanced, cond_idx
            )
            prev_cond = self.conditional.mix_by_mask(cur_cond, prev_cond, is_new)
            clogits = clogits.log_softmax(1).masked_fill(~is_new.unsqueeze(1), 0)
            cur_tok = cond_hist.gather(0, cond_idx.unsqueeze(0)).squeeze(0).clamp_(0, V)
            ll = ll + clogits.gather(1, cur_tok.unsqueeze(1)).squeeze(1)
            cond_idx = (cond_idx + latent_hist[t]).clamp_max_(L - 1)
        return ll.masked_fill(cond_lens > L, torch.finfo(torch.float).min)


class LstmLmParams(param.Parameterized):

    embedding_size = param.Integer(
        0, bounds=(0, None), doc="The size of token embeddings."
    )
    hidden_size = param.Integer(
        300, bounds=(1, None), doc="The size of the LSTM's hidden states."
    )
    num_layers = param.Integer(3, bounds=(0, None), doc="The number of LSTM layers.")
    merge_method = param.ObjectSelector(
        "mix",
        ["mix", "cat"],
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
    length_name: str
    lstm_input_size: int
    logiter_input_size: int

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
        length_name: str = "length",
    ):
        if params is None:
            params = LstmLmParams()
        # FIXME(sdrobert): bad coding
        self._params = params
        using_names = [
            input_name,
            post_input_name,
            hidden_name,
            cell_name,
            last_hidden_name,
            length_name,
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
        self.length_name = length_name
        if params.embedding_size > 0:
            self.embedder = torch.nn.Embedding(
                vocab_size + 1, params.embedding_size, vocab_size
            )
        else:
            self.add_module("embedder", None)
        if (
            input_size > 0
            and params.embedding_size > 0
            and params.merge_method == "mix"
        ):
            self.input_merger = torch.nn.Linear(input_size, params.embedding_size)
            self.lstm_input_size = params.embedding_size
        else:
            self.lstm_input_size = params.embedding_size + input_size
            self.add_module("input_merger", None)
        self.dropout = torch.nn.Dropout(0.0)
        if self.lstm_input_size == 0 and (
            params.num_layers or post_input_size == 0 or params.merge_method == "mix"
        ):
            raise ValueError("Model input dim is 0")
        if params.num_layers:
            self.lstm = torch.nn.LSTM(
                self.lstm_input_size, params.hidden_size, params.num_layers
            )
        else:
            self.add_module("lstm", None)
        # We postpone initialization of the cells to the end of init to
        # maintain consistency with reset_parameters.
        self.logiter_input_size = (
            params.hidden_size if params.num_layers else self.lstm_input_size
        )
        if post_input_size > 0:
            if params.merge_method == "mix":
                self.post_merger = torch.nn.Linear(
                    post_input_size, self.logiter_input_size
                )
            else:
                self.logiter_input_size += post_input_size
                self.add_module("post_merger", None)
        else:
            self.add_module("post_merger", None)
        self.logiter = torch.nn.Linear(self.logiter_input_size, vocab_size)
        in_size = self.lstm_input_size
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
        past_length = torch.full((vocab_size,), config.EPS_NINF)
        past_length[0] = config.EPS_0
        self.register_buffer("past_length", past_length, persistent=False)

    @property
    def dropout_prob(self) -> float:
        return self.dropout.p

    @dropout_prob.setter
    def dropout_prob(self, val: float):
        self.dropout.p = val

    def reset_parameters(self) -> None:
        if self.embedder is not None:
            self.embedder.reset_parameters()
        if self.input_merger is not None:
            self.input_merger.reset_parameters()
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.post_merger is not None:
            self.post_merger.reset_parameters()
        self.logiter.reset_parameters()
        # don't reset cells b/c they share parameters w/ LSTM

    def update_input(self, prev: StateDict, hist: torch.Tensor) -> StateDict:
        N, L = hist.size(1), len(self.cells)
        H = self.lstm.weight_hh_l0.size(1) if L else self.lstm_input_size
        if self.hidden_name not in prev:
            prev[self.hidden_name] = torch.zeros(L, N, H, device=hist.device)
        if self.cell_name not in prev:
            prev[self.cell_name] = torch.zeros(L, N, H, device=hist.device)
        if self.last_hidden_name not in prev:
            prev[self.last_hidden_name] = torch.zeros(N, H, device=hist.device)
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
        if self.length_name in prev:
            new_prev[self.length_name] = prev[self.length_name].index_select(0, src)
        return new_prev

    def mix_by_mask(
        self, prev_true: StateDict, prev_false: StateDict, mask: torch.Tensor
    ) -> StateDict:
        mask = mask.float()
        mask_2, mask_3 = mask.view(-1, 1), mask.view(1, -1, 1)
        prev = {
            self.hidden_name: mask_3 * prev_true[self.hidden_name]
            + (1 - mask_3) * prev_false[self.hidden_name],
            self.cell_name: mask_3 * prev_true[self.cell_name]
            + (1 - mask_3) * prev_false[self.cell_name],
            self.last_hidden_name: mask_2 * prev_true[self.last_hidden_name]
            + (1 - mask_2) * prev_false[self.last_hidden_name],
        }
        if self.input_name in prev_true and self.input_size:
            x = prev_true[self.input_name]
            assert x.dim() == 3
            prev[self.input_name] = x
        if self.post_input_name in prev_true and self.post_input_size:
            x = prev_true[self.post_input_name]
            assert x.dim() == 3
            prev[self.post_input_name] = x
        if self.length_name in prev_true:
            prev[self.length_name] = prev_true[self.length_name]
        return prev

    @staticmethod
    def mix_vectors(a, b):
        w = (a - b).sigmoid()
        return a * w + b * (1 - w)

    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: StateDict, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, StateDict]:
        T, N = hist.shape
        V, L = self.vocab_size, len(self.cells)
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
        if self.lstm_input_size:
            if in_ is None or in_.size(1) != self.lstm_input_size:
                raise RuntimeError("Input missing or of incorrect size")
        else:
            in_ = prev[self.last_hidden_name]
        prev_hidden, prev_cell = prev[self.hidden_name], prev[self.cell_name]
        cur_hidden, cur_cell = [], []
        for lidx in range(L):
            if lidx:
                in_ = self.dropout(in_)
            hidden, cell = self.cells[lidx](in_, (prev_hidden[lidx], prev_cell[lidx]))
            cur_hidden.append(hidden)
            cur_cell.append(cell)
            in_ = hidden
        if L:
            cur[self.hidden_name] = torch.stack(cur_hidden)
            cur[self.cell_name] = torch.stack(cur_cell)
        else:
            cur[self.hidden_name] = cur[self.cell_name] = prev[self.hidden_name]
        cur[self.last_hidden_name] = in_
        if self.post_input_name in prev and self.post_input_size:
            x = prev[self.post_input_name]
            if x.dim() != 2:
                cur[self.post_input_name] = x
                x = x.gather(
                    0, idx.unsqueeze(2).expand(1, N, self.post_input_size)
                ).squeeze(0)
            if self.post_merger is not None:
                x = self.post_merger(x)
                in_ = self.mix_vectors(in_, x)
            else:
                in_ = torch.cat([in_, x], 1)
        logits = self.logiter(in_)
        if self.length_name in prev:
            length = prev[self.length_name]
            cur[self.length_name] = length
            mask = (idx.squeeze(0) >= length).unsqueeze(1)
            logits = torch.where(mask, self.past_length.expand_as(logits), logits)
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
        if self.lstm_input_size:
            if in_ is None or in_.size(2) != self.lstm_input_size:
                raise RuntimeError("Input missing or of incorrect size")
        else:
            in_ = prev[self.last_hidden_name].unsqueeze(0)
        if self.lstm is not None:
            self.lstm.dropout = self.dropout.p
            in_ = self.lstm(in_, (prev[self.hidden_name], prev[self.cell_name]))[0]
        return in_

    def calc_all_logits(self, prev: StateDict, hidden: torch.Tensor) -> torch.Tensor:
        if self.post_input_name in prev and self.post_input_size:
            x = prev[self.post_input_name]
            if x.dim() != 3:
                raise RuntimeError(
                    f"full post input ('{self.post_input_name}') must be provided"
                )
            if self.post_merger is not None:
                x = self.post_merger(x)
                hidden = self.mix_vectors(hidden, x)
            else:
                hidden = x if hidden is None else torch.cat([hidden, x], 2)
        logits = self.logiter(hidden)
        if self.length_name in prev:
            length = prev[self.length_name]
            mask = (
                torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
                >= length
            ).unsqueeze(2)
            logits = torch.where(mask, self.past_length.expand_as(logits), logits)
        return logits

    def calc_full_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.calc_all_logits(prev, self.calc_all_hidden(prev, hist))


class JointLatentLstmLm(JointLatentLanguageModel):

    latent: LstmLm
    conditional: LstmLm

    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        latent_params: Optional[LstmLmParams] = None,
        cond_params: Optional[LstmLmParams] = None,
        cond_input_is_post: bool = False,
        input_name: str = "input",
        latent_prefix: str = "latent_",
        conditional_prefix: str = "conditional_",
        old_conditional_prefix: str = "old_",
    ):
        if latent_params is None:
            latent_params = LstmLmParams()
        if cond_params is None:
            cond_params = latent_params
        latent = LstmLm(2, input_size, params=latent_params, input_name=input_name)
        in_size = latent.logiter_input_size
        if cond_input_is_post:
            conditional = LstmLm(
                vocab_size,
                post_input_size=in_size,
                params=cond_params,
                post_input_name=latent_prefix + "last",
            )
        else:
            conditional = LstmLm(
                vocab_size,
                input_size=in_size,
                params=cond_params,
                input_name=latent_prefix + "last",
            )
        super().__init__(
            latent,
            conditional,
            latent_prefix,
            conditional_prefix,
            old_conditional_prefix,
        )

    def calc_full_log_probs(self, hist: torch.Tensor, prev: StateDict) -> torch.Tensor:
        T, N = hist.shape
        V = self.conditional.vocab_size
        prev_latent, prev_cond, _, _ = self.split_dicts(prev)
        latent_hist = hist != V
        latent_hist_ = latent_hist.long()
        hidden = self.latent.calc_all_hidden(prev_latent, latent_hist_)
        llogits = self.latent.calc_all_logits(prev_latent, hidden)
        if T > 0:
            hist = extended_hist_to_conditional(hist, latent_hist=latent_hist)[0]
            cond_idx = latent_hist_.T.cumsum(1)
            cond_idx = torch.cat([cond_idx.new_zeros(1, N), cond_idx.T], 0)
        else:
            cond_idx = hist.new_zeros((1,))
        cond_idx = cond_idx.expand(T + 1, N)
        clogits = []
        old_cond = prev_cond.copy()
        for t in range(T + 1):
            is_new = latent_hist[max(t - 1, 0)] & (t > 0)
            prev_cond = self.conditional.mix_by_mask(prev_cond, old_cond, is_new)
            # don't pass input all at once! idx != cond_idx
            prev_cond[self.conditional.input_name] = hidden[t]
            prev_cond[self.conditional.post_input_name] = hidden[t]
            clogits_t, cur_cond = self.conditional.calc_idx_log_probs(
                hist, prev_cond, cond_idx[t]
            )
            clogits.append(clogits_t)
            prev_cond, old_cond = cur_cond, prev_cond
        clogits = torch.stack(clogits).log_softmax(2)
        logits = torch.cat([llogits[..., 1:] + clogits, llogits[..., :1]], 2)
        return logits

    def calc_log_likelihood_given_latents(
        self,
        latent_hist: torch.Tensor,
        cond_hist: torch.Tensor,
        prev: StateDict = dict(),
    ) -> torch.Tensor:
        # we're in a better position than when calculating the log probabilities b/c
        # we don't need to worry about the rest of the pdf when the latent variable
        # is off.
        T, N = latent_hist.shape
        L, N_ = cond_hist.shape
        assert N == N_
        latent_hist_ = latent_hist.T.bool()
        cond_lens = latent_hist.sum(0)
        Lmax = int(cond_lens.max().item())
        if Lmax > L:
            cond_hist = torch.nn.functional.pad(cond_hist, (0, 0, 0, Lmax - L))
        else:
            Lmax = L
        prev = self.update_input(prev.copy(), latent_hist)
        prev_latent, prev_cond, _, _ = self.split_dicts(prev)
        hidden = self.latent.calc_all_hidden(prev_latent, latent_hist[:-1])

        H = hidden.size(2)
        assert hidden.shape == (T, N, H)
        hidden = hidden.transpose(0, 1)
        hidden_mask = cond_lens.unsqueeze(1) > torch.arange(Lmax, device=hidden.device)
        hidden_ = hidden.masked_select(latent_hist_.unsqueeze(2).expand(N, T, H))
        hidden = hidden.new_zeros((N, Lmax, H)).masked_scatter(
            hidden_mask.unsqueeze(2).expand(N, Lmax, H), hidden_
        )
        hidden = hidden.transpose(0, 1)
        prev_cond[self.conditional.input_name] = hidden
        prev_cond[self.conditional.post_input_name] = hidden
        cond_hist = cond_hist.clamp(0, self.vocab_size - 1)
        hidden = self.conditional.calc_all_hidden(prev_cond, cond_hist[:-1])
        clogits = self.conditional.calc_all_logits(prev_cond, hidden).log_softmax(2)
        ll = clogits.gather(2, cond_hist.unsqueeze(2)).squeeze(2)
        ll = ll.masked_fill(~hidden_mask.T, 0).sum(0)
        return ll.masked_fill(cond_lens > L, torch.finfo(torch.float).min)

    def calc_marginal_log_likelihoods(
        self,
        cond_hist: torch.Tensor,
        given_count: torch.Tensor,
        prev: StateDict = dict(),
    ) -> torch.Tensor:
        if self.conditional.input_size > 0 or self.latent.embedder is not None:
            raise NotImplementedError
        assert self.conditional.post_input_size
        prev = self.update_input(prev, cond_hist)
        prev_latent, prev_cond, _, _ = self.split_dicts(prev)
        cond_hist = cond_hist.clamp(0, self.vocab_size - 1)

        lhidden = self.latent.calc_all_hidden(prev_latent)  # cond_hist is cond input
        T, N, H1 = lhidden.shape
        llogits = self.latent.calc_all_logits(prev_latent, lhidden)
        llogits = llogits.transpose(0, 1)[..., 1] - llogits.transpose(0, 1)[..., 0]
        dist = distributions.ConditionalBernoulli(given_count, logits=llogits)

        L = cond_hist.size(0)
        chidden = self.conditional.calc_all_hidden(prev_cond, cond_hist[:-1])
        H2 = chidden.size(2)
        chidden = chidden.unsqueeze(1).expand(L, T, N, H2)
        if self.conditional.post_merger is not None:
            lhidden = self.conditional.post_merger(lhidden)
            lhidden = lhidden.expand(L, T, N, H2)
            jhidden = self.conditional.mix_vectors(chidden, lhidden)
        else:
            lhidden = lhidden.expand(L, T, N, H1)
            jhidden = torch.cat([chidden, lhidden], 3)
        clogits = self.conditional.logiter(jhidden).log_softmax(3)  # (L, T, N, V)
        lcond = clogits.gather(
            3, cond_hist.view(L, 1, N, 1).expand(L, T, N, 1)
        ).squeeze(3)

        ll = dist.marginal_log_likelihoods(lcond.transpose(1, 2), False)
        denom = llogits.exp().log1p().sum(1)
        ll = ll - denom
        assert (ll < 0).all()
        return ll


## TESTS


def test_suffix_forcing():
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
    walk = RandomWalk(uniform_binary)
    dist = SequentialLanguageModelDistribution(walk, N, max_iters=T)
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
    suffix_forcing = SuffixForcingWrapper(uniform_binary)
    walk = RandomWalk(suffix_forcing)
    dist = SequentialLanguageModelDistribution(
        walk, N, {"total": total, "given": given}, max_iters=T
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
    walk = RandomWalk(joint)
    dist = SequentialLanguageModelDistribution(walk, max_iters=T)
    support = dist.enumerate_support()
    assert support.shape == ((V + 1) ** T, T)
    assert (support == V).any()
    log_probs = dist.log_prob(support)
    assert torch.allclose(log_probs.logsumexp(0), torch.tensor(0.0))


def test_lstm_lm():
    torch.manual_seed(3)

    T, N, V, H = 10, 3, 5, 20
    loss_fn = torch.nn.CrossEntropyLoss()
    for merge_method in ("mix", "cat"):
        for embedding_size, input_size in itertools.product(range(0, 10, 5), repeat=2):
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
                length = torch.randint(T + 1, (N,))
                prev = {
                    input_name: torch.randn(T, N, V),
                    "length": length,
                }
                lm = LstmLm(V, input_size, post_input_size, params)
                logits_exp = lm(hist[:-1], prev)
                for n in range(N):
                    assert (logits_exp[length[n] :, n, 1:] == config.EPS_NINF).all()
                    assert (logits_exp[length[n] :, n, 0] == config.EPS_0).all()
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
    import pydrobert.torch.estimators as _e
    import pydrobert.torch.distributions as _d
    import pydrobert.torch.modules as _m

    torch.manual_seed(4)

    T, N, V, H, I, M = 6, 7, 3, 4, 5, 2 ** 12
    hist = torch.randint(V + 1, (T, N))
    in_ = torch.rand(T, N, I)
    loss_fn = torch.nn.CrossEntropyLoss()
    prev = {"latent_input": in_, "latent_lens": torch.randint(T + 1, (N,))}
    for merge_method in ("cat", "mix"):
        for input_post in (False, True):
            for num_layers, embedding_size in itertools.product((0, 3), repeat=2):
                if input_post and embedding_size == 0:
                    continue
                print(
                    f"num_layers={num_layers}, embedding_size={embedding_size}, "
                    f"input_post={input_post}, merge_method={merge_method}"
                )
                params = LstmLmParams(
                    embedding_size=embedding_size,
                    hidden_size=H,
                    num_layers=num_layers,
                    merge_method=merge_method,
                )
                lm_act = JointLatentLstmLm(V, I, params, cond_input_is_post=input_post)
                latent = lm_act.latent
                conditional = lm_act.conditional
                lm_exp = JointLatentLanguageModel(latent, conditional)

                logits_exp = lm_exp(hist[:-1], prev)
                assert logits_exp.shape == (T, N, V + 1)
                loss_exp = loss_fn(logits_exp.flatten(end_dim=-2), hist.flatten())
                g_exp = torch.autograd.grad(loss_exp, lm_exp.parameters())

                logits_act = lm_act(hist[:-1], prev)
                assert logits_act.shape == (T, N, V + 1)
                assert torch.allclose(logits_exp, logits_act)
                loss_act = loss_fn(logits_act.flatten(end_dim=-2), hist.flatten())
                assert torch.allclose(loss_exp, loss_act)
                g_act = torch.autograd.grad(loss_act, lm_act.parameters())
                for g_exp_p, g_act_p in zip(g_exp, g_act):
                    assert torch.allclose(g_exp_p, g_act_p)

                latent_hist = torch.randint(2, (T, N))
                cond_hist = torch.randint(V, (T // 2, N))
                ll_exp = lm_exp.calc_log_likelihood_given_latents(
                    latent_hist, cond_hist, prev
                )
                assert ll_exp.shape == (N,)
                ll_act = lm_act.calc_log_likelihood_given_latents(
                    latent_hist, cond_hist, prev
                )
                assert torch.allclose(ll_exp, ll_act)

                if not input_post or not embedding_size:
                    continue

                latent_params = LstmLmParams(
                    hidden_size=H, num_layers=num_layers, merge_method=merge_method
                )
                cond_params = params

                lm = JointLatentLstmLm(
                    V, I, latent_params, cond_params, cond_input_is_post=True
                )
                tok = torch.randint(V, (T // 2, N))
                given_count = torch.randint(1, T // 2 + 1, (N,))
                walk = _m.RandomWalk(lm.latent)
                dist = _d.SequentialLanguageModelDistribution(
                    walk, N, {"input": in_}, cache_samples=True, max_iters=T
                )
                proposal = _d.SimpleRandomSamplingWithoutReplacement(given_count, T)

                def func(b):
                    M_ = b.size(0)
                    bad_count = b.sum(-1) != given_count
                    # assert not bad_count.any()
                    ll = lm.calc_log_likelihood_given_latents(
                        b.flatten(end_dim=-2).T.long(),
                        tok.unsqueeze(1).expand(-1, M_, -1).flatten(1),
                        {
                            "latent_input": in_.unsqueeze(1)
                            .expand(-1, M_, -1, -1)
                            .flatten(1, -2)
                        },
                    )
                    return ll.view(b.shape[:-1]).masked_fill(bad_count, config.EPS_NINF)

                # estimator = _e.DirectEstimator(dist, func, M, is_log=True)
                estimator = _e.ImportanceSamplingEstimator(
                    proposal, func, M, dist, is_log=True
                )
                act = estimator()

                exp = lm.calc_marginal_log_likelihoods(tok, given_count, prev)
                assert torch.allclose(exp, act, atol=1e-2)
