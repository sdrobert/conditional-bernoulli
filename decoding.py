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

"""Decoding modules"""

from typing import Dict, Optional, Tuple, overload

import torch

from models import JointLatentLanguageModel


class JointLatentLanguageModelPrefixSearch(torch.nn.Module):

    lm: JointLatentLanguageModel
    width: int
    normalize: bool

    def __init__(
        self, lm: JointLatentLanguageModel, width: int, normalize: bool = True
    ):
        super().__init__()
        self.lm = lm
        self.width = width
        self.normalize = normalize

    def reset_parameters(self):
        self.lm.reset_parameters()

    @overload
    def forward(
        self, batch_size: int, max_iters: int, prev: Dict[str, torch.Tensor] = dict()
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @torch.no_grad()
    def forward(
        self, N: int, T: int, prev_: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev = dict() if prev_ is None else prev_
        device = next(iter(self.lm.parameters())).device
        V = self.lm.conditional.vocab_size

        probs_prev = torch.ones((N, 1), device=device, dtype=torch.float)
        y_prev = torch.full((T, N, 1), V, device=device, dtype=torch.long)
        y_prev_cond = torch.zeros((T, N, 1), device=device, dtype=torch.long)
        y_prev_lens = torch.zeros((N, 1), dtype=torch.long, device=device)
        prev_is_prefix = torch.ones((N, 1, 1), device=device, dtype=torch.bool)
        prev = self.lm.update_input(prev, y_prev.flatten(1))
        Kp = 1

        for t in torch.arange(T, device=device):
            K = min(self.width, Kp * (V + 1))
            lprobs_t, cand = self.lm.calc_idx_log_probs(y_prev.flatten(1), prev, t)
            probs_t = lprobs_t.unflatten(0, (N, Kp)).softmax(2)
            ext_probs_t, nonext_probs_t = probs_t[..., :-1], probs_t[..., -1]
            ext_probs_cand = ext_probs_t * probs_prev.unsqueeze(2)  # (N, Kp, V)
            nonext_probs_cand = nonext_probs_t * probs_prev  # (N, Kp)

            # determine which extending candidates end up matching a non-extending
            # candidate and move the entire probability mass to the latter
            # (N, Kp-prefix, Kp-parent)
            tok_to_match = y_prev_cond.gather(
                0, y_prev_lens.unsqueeze(2).expand(N, Kp, Kp).transpose(0, 1)
            ).transpose(
                0, 1
            )  # (N, Kp, Kp)
            ext_is_exact = (
                (y_prev_lens + 1).unsqueeze(2) == y_prev_lens.unsqueeze(1)
            ) & prev_is_prefix  # (N, Kp, Kp)
            has_match = (
                torch.nn.functional.one_hot(tok_to_match, V).bool()
                & ext_is_exact.unsqueeze(-1)
            ).any(
                2
            )  # (N, Kp, V)
            nonext_probs_cand += (
                ext_probs_cand.gather(2, tok_to_match)
                .masked_fill(~ext_is_exact, 0)
                .sum(1)
            )
            ext_probs_cand.masked_fill_(has_match, -float("inf"))

            # place the non-extending candidates after the extending candidates
            # and determine the top K
            probs_cand = torch.cat([ext_probs_cand.flatten(1), nonext_probs_cand], 1)
            probs_next, next_idx = probs_cand.topk(K)
            next_is_nonext = next_idx >= Kp * V
            next_src = torch.where(
                next_is_nonext,
                next_idx - (Kp * V),
                next_idx.div(V, rounding_mode="trunc"),
            )
            next_ext = next_idx % V

            y_next_prefix_lens = y_prev_lens.gather(1, next_src)
            y_next_cond = y_prev_cond.gather(
                2, next_src.unsqueeze(0).expand(T, N, K)
            ).scatter_(0, y_next_prefix_lens.unsqueeze(0), next_ext.unsqueeze(0))
            y_next = y_prev.gather(2, next_src.unsqueeze(0).expand(T, N, K))
            y_next[t] = torch.where(
                next_is_nonext, torch.full_like(next_ext, V), next_ext
            )
            y_next_lens = y_next_prefix_lens + ~next_is_nonext

            next_prefix_is_prefix = prev_is_prefix.gather(
                1, next_src.unsqueeze(2).expand(N, K, Kp)
            ).gather(2, next_src.unsqueeze(1).expand(N, K, K))
            next_len_leq = y_next_lens.unsqueeze(2) <= y_next_lens.unsqueeze(1)
            next_to_match = y_next_cond.gather(
                0,
                (y_next_lens - 1)
                .clamp(min=0)
                .unsqueeze(2)
                .expand(N, K, K)
                .transpose(0, 1),
            ).transpose(0, 1)
            next_ext_matches = next_to_match == next_ext.unsqueeze(2)
            next_is_prefix = (
                next_prefix_is_prefix
                & next_len_leq
                & (
                    next_is_nonext.unsqueeze(2)
                    | (~next_is_nonext.unsqueeze(2) & next_ext_matches)
                )
            )

            if K < self.width:
                rem = self.width - K
                neg_inf = torch.full((N, rem), -float("inf"), device=device)
                zeros = next_src.new_zeros(N, rem)
                y_next = torch.cat([y_next, y_next.new_full((T, N, rem), V)], 2)
                y_next_cond = torch.cat(
                    [y_next_cond, y_next_cond.new_zeros(T, N, rem)], 2
                )
                y_next_lens = torch.cat([y_next_lens, zeros], 1)
                probs_next = torch.cat([probs_next, neg_inf], 1)
                false_ = torch.zeros((N, rem), device=device, dtype=torch.bool)
                next_is_nonext = torch.cat([next_is_nonext, false_], 1)
                next_is_prefix = torch.cat(
                    [next_is_prefix, false_.unsqueeze(1).expand(N, K, rem)], 2
                )
                next_is_prefix = torch.cat(
                    [next_is_prefix, false_.unsqueeze(2).expand(N, rem, self.width)], 1
                )
                next_src = torch.cat([next_src, zeros], 1)
                K = self.width

            next_ = self.lm.extract_by_src(cand, next_src.flatten())

            if self.normalize:
                probs_next /= probs_next.max(1, keepdim=True)[0]

            prev, probs_prev, y_prev_cond = next_, probs_next, y_next_cond
            y_prev, y_prev_lens, prev_is_prefix = y_next, y_next_lens, next_is_prefix
            Kp = K

        return y_prev_cond, y_prev_lens, probs_prev


def test_joint_latent_language_model_prefix_search():
    import models

    torch.manual_seed(1)

    T, N, V, H, I, E = 4, 4, 5, 3, 5, 5
    W = (V + 1) ** T

    cond_hist = torch.randint(V, (T, N))
    in_ = torch.randn(T, N, I)
    latent_lens = torch.randint(T + 1, (N,))
    latent_lens[0] = T
    given_count = (torch.rand(N) * latent_lens).long()
    prev = {"latent_input": in_, "latent_lens": latent_lens}
    latent_params = models.LstmLmParams(embedding_size=0, hidden_size=H)
    cond_params = models.LstmLmParams(embedding_size=E, hidden_size=H)
    lm = models.JointLatentLstmLm(
        V, I, latent_params, cond_params, cond_input_is_post=True
    )
    exp = lm.calc_marginal_log_likelihoods(cond_hist, given_count, prev)
    len_mask = torch.arange(T).unsqueeze(1) >= given_count
    cond_hist.masked_fill_(len_mask, -1)

    search = JointLatentLanguageModelPrefixSearch(lm, W, normalize=False)
    beam, beam_lens, act = search(N, T, prev)
    invalid_mask = (act <= 0).expand_as(beam)
    len_mask = torch.arange(T).view(T, 1, 1) >= beam_lens
    beam.masked_fill_(len_mask, -1)
    beam.masked_fill_(invalid_mask, -2)
    seq_match = (beam == cond_hist.unsqueeze(2)).all(0)
    assert (seq_match.sum(1) == 1).all()
    act = act.masked_fill_(~seq_match, 0.0).sum(1).log_()
    assert torch.allclose(exp, act, atol=1e-1)

    search.normalize = True
    beam2, _, _ = search(N, T, prev)
    beam2.masked_fill_(len_mask, -1)
    beam2.masked_fill_(invalid_mask, -2)
    assert (beam == beam2).all()
