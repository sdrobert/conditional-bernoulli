import sys
import argparse
import os
import math
import glob
import gzip

from typing import Optional, Tuple
from typing_extensions import Literal

import torch
import param
import pydrobert.torch.config as config

config.USE_JIT = True

import pydrobert.torch.data as data
import pydrobert.torch.training as training
import pydrobert.torch.modules as modules
import pydrobert.torch.estimators as pestimators
import pydrobert.torch.distributions as pdistributions
import pydrobert.param.argparse as pargparse

from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record


import models
import distributions
import estimators
import decoding


def get_filts_and_classes(train_dir: str) -> Tuple[int, int]:
    """Given the training partition directory, determine the number of filters/classes
    Always use training partition info! Number of filts in test partition might be the
    same, but maybe not the number of classes.
    Returns
    -------
    num_filts, num_classes : int, int
    """
    part_name = os.path.basename(train_dir)
    ext_file = os.path.join(os.path.dirname(train_dir), "ext", f"{part_name}.info.ark")
    if not os.path.isfile(ext_file):
        raise ValueError(f"Could not find '{ext_file}'")
    dict_ = dict()
    with open(ext_file) as file_:
        for line in file_:
            k, v = line.strip().split()
            dict_[k] = v
    return int(dict_["num_filts"]), int(dict_["max_ref_class"]) + 1


def get_num_avail_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count()


class MergeParams(param.Parameterized):

    cond_input_is_post = param.Boolean(False)


class AcousticModelTrainingParams(
    training.TrainingStateParams, data.SpectDataLoaderParams
):
    estimator = param.ObjectSelector(
        "direct",
        objects=[
            "direct",
            "marginal",
            "cb",
            "srswor",
            "ais-c",
            "ais-g",
            "ais-s",
            "sf-biased",
            "sf-is",
            "ctc",
        ],
    )
    mc_samples = param.Integer(1, bounds=(1, None))
    mc_burn_in = param.Integer(1, bounds=(1, None))
    dropout_prob = param.Magnitude(0.0)
    swap_prob = param.Magnitude(0.0)
    aug_er_thresh = param.Magnitude(0.0)
    sa_time_size_prop = param.Magnitude(0.0)
    sa_time_num_prop = param.Magnitude(0.0)
    sa_freq_num = param.Integer(0, bounds=(0, None))
    sa_freq_size = param.Integer(0, bounds=(0, None))
    weight_noise_std = param.Number(0, bounds=(0, None))
    optimizer = param.ObjectSelector("sgd", ["adam", "sgd"])
    momentum = param.Magnitude(0.0)
    entropy_decay = param.Magnitude(0.97 ** (1 / 10000))
    entropy_floor = param.Magnitude(0.1)
    entropy_init = param.Magnitude(0.0)
    world_size = param.Integer(1, bounds=(1, None))
    centering = param.Boolean(False)


class LanguageModelTrainingParams(
    training.TrainingStateParams, data.LangDataLoaderParams
):
    fraction_held_out = param.Magnitude(0.05)
    aug_pp_thresh = param.Number(0.0, bounds=(0, None))
    dropout_prob = param.Magnitude(0.0)
    swap_prob = param.Magnitude(0.0)
    weight_noise_std = param.Number(0, bounds=(0, None))
    optimizer = param.ObjectSelector("sgd", ["adam", "sgd"])
    momentum = param.Magnitude(0.0)


class DecodingParams(param.Parameterized):
    style = param.ObjectSelector("beam", objects=["beam", "prefix", "ctc"])
    is_ctc = param.Boolean(False)


def construct_default_param_dict():
    return {
        "model": {
            "frontend": models.FrontendParams(name="model.frontend"),
            "latent": models.LstmLmParams(name="model.latent"),
            "conditional": models.LstmLmParams(name="model.conditional"),
            "merge": MergeParams(name="model.merge"),
        },
        "am": {
            "training": AcousticModelTrainingParams(name="am.training"),
            "decoding": DecodingParams(name="am.decoding"),
            "pcb": models.FrontendParams(name="am.pcb"),
        },
        "lm": LanguageModelTrainingParams(name="lm.training"),
    }


def train_lm_for_epoch(
    model: models.LstmLm,
    loader: data.LangDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    params: LanguageModelTrainingParams,
    epoch: int,
    device: torch.device,
    quiet: int,
) -> float:
    loader.epoch = epoch
    non_blocking = False  # device.type == "cpu" or loader.pin_memory
    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.INDEX_PAD_VALUE)
    wn_setup = wn_teardown = lambda: None
    do_aug = controller.get_info(epoch - 1)["do_aug"]
    if do_aug:
        model.dropout_prob = params.dropout_prob
        model.swap_prob = params.swap_prob
        if params.weight_noise_std:
            param_data = []

            def wn_setup():
                for p in model.parameters():
                    param_data.append(p.data.clone())
                    p.data += torch.randn_like(p) * params.weight_noise_std

            def wn_teardown():
                for p, p_ in zip(model.parameters(), param_data):
                    # we copy so that RNN parameters remain contiguous in memory
                    p.data.copy_(p_)
                param_data.clear()

    model.train()

    num_samples = len(loader.batch_sampler.sampler)

    if quiet < 1:
        loader = tqdm(loader)

    total_loss = 0
    for hyp, _ in loader:
        hyp = hyp.to(device, non_blocking=non_blocking)
        hist = hyp[:-1].clamp(0, model.vocab_size - 1)

        optimizer.zero_grad()
        wn_setup()
        logits = model(hist)
        assert logits.shape[:-1] == hyp.shape
        loss = loss_fn(logits.flatten(0, 1), hyp.flatten())
        loss.backward()
        wn_teardown()
        optimizer.step()
        total_loss += loss.detach() * hyp.size(1)

    return total_loss.item() / num_samples


@torch.no_grad()
def lm_perplexity(
    model: models.SequentialLanguageModel, val: data.LangDataSet, device: torch.device,
) -> float:

    model.eval()

    total_ll = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for hyp in val:
        hyp = hyp.to(device)
        logits = model(hyp[:-1].unsqueeze(1)).squeeze(1)
        total_ll -= loss_fn(logits, hyp).item()
        total_tokens += hyp.numel()

    return math.exp(-total_ll / max(total_tokens, 1))


def init_model(options, dict_, delta_order, pretraining=False) -> models.AcousticModel:
    device = options.device
    num_filts, num_classes = get_filts_and_classes(
        os.path.join(options.data_dir, "train")
    )
    dict_conditional = dict_["conditional"]
    cond_input_is_post = dict_["merge"].cond_input_is_post
    if pretraining:
        num_classes = 1
        dict_conditional = models.LstmLmParams(embedding_size=0, num_layers=0)
        cond_input_is_post = True

    model = models.AcousticModel(
        num_classes,
        num_filts * (delta_order + 1),
        dict_["frontend"],
        dict_["latent"],
        dict_conditional,
        cond_input_is_post,
    )
    if pretraining:
        for p in model.conditional.parameters():
            p.data.fill_(1.0)
            p.requires_grad_(False)
    if "pretrained_enc_path" in options and options.pretrained_enc_path is not None:
        state_dict = torch.load(options.pretrained_enc_path)
        assert not any(k.startswith("conditional.") for k in state_dict)
        model.load_state_dict(state_dict, strict=False)
        # don't let the controller reset the parameters on the first epoch!
        model.frontend.reset_parameters = (
            model.latent.reset_parameters
        ) = lambda *args, **kwargs: None
    if "pretrained_lm_path" in options and options.pretrained_lm_path is not None:
        assert not options.pretraining
        state_dict = torch.load(options.pretrained_lm_path)
        model.conditional.load_state_dict(state_dict, strict=False)
        # for key, param in model.conditional.named_parameters():
        #     if state_dict.get(key, None) is not None:
        #         param.requires_grad = False
        # don't let the controller reset the parameters on the first epoch!
        model.conditional.reset_parameters = lambda *args, **kwargs: None
    if "am_path" in options:
        assert options.am_path is not None
        state_dict = torch.load(options.am_path)
        model.load_state_dict(state_dict)
    return model.to(device)


def init_pcb(options, frontend_params, delta_order) -> models.AcousticModel:
    device = options.device
    num_filts, num_classes = get_filts_and_classes(
        os.path.join(options.data_dir, "train")
    )
    if options.pcb_path is None:
        raise ValueError("pcb estimator chosen but --pcb-path not specified")
    state_dict = torch.load(options.pcb_path)
    for k in tuple(state_dict):
        if k.startswith("conditional."):
            state_dict.remove(k)
    other_params = models.LstmLmParams(embedding_size=0, num_layers=0)
    model = models.AcousticModel(
        num_classes,
        num_filts * (delta_order + 1),
        frontend_params,
        other_params,
        other_params,
        True,
    )
    model.load_state_dict(state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    return model.eval().to(device)


def train_lm(options, dict_):
    os.makedirs(options.model_dir, exist_ok=True)
    train_data_dir = os.path.join(options.data_dir, "lm")
    if not os.path.isdir(train_data_dir):
        raise ValueError(
            f"'{train_data_dir}' is not a directory. Did you initialize it?"
        )
    dev_data_dir = os.path.join(options.data_dir, "dev", "ref")
    if not os.path.isdir(dev_data_dir):
        raise ValueError(f"'{dev_data_dir}' is not a directory. Did you initialize it?")
    seed = dict_["lm"].seed

    model = init_model(options, dict_["model"], dict_["am"]["training"].delta_order)
    model = model.conditional
    model.add_module("post_merger", None)
    model.add_module("input_merger", None)
    if dict_["lm"].optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 1e-4, momentum=dict_["lm"].momentum
        )
    elif dict_["lm"].optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    else:
        assert False

    state_csv = os.path.join(options.model_dir, "hist.csv")
    state_dir = os.path.join(
        options.model_dir if options.ckpt_dir is None else options.ckpt_dir, "training"
    )
    os.makedirs(state_dir, exist_ok=True)

    controller = training.TrainingStateController(
        dict_["lm"], state_csv, state_dir, warn=options.quiet < 1
    )
    controller.add_entry("do_aug", int)

    loader = data.LangDataLoader(
        train_data_dir,
        dict_["lm"],
        shuffle=True,
        seed=seed,
        batch_first=False,
        num_workers=min(max(get_num_avail_cores() - 1, 1), 4),
    )

    val = data.LangDataSet(dev_data_dir, dict_["lm"])

    epoch = controller.get_last_epoch() + 1
    do_aug = bool(controller.get_info(epoch - 1)["do_aug"])
    aug_pp_thresh = dict_["lm"].aug_pp_thresh
    while controller.continue_training(epoch - 1):
        if options.quiet < 1:
            print(f"Training epoch {epoch}...", file=sys.stderr)
        train_loss = train_lm_for_epoch(
            model,
            loader,
            optimizer,
            controller,
            dict_["lm"],
            epoch,
            options.device,
            options.quiet,
        )
        if options.quiet < 1:
            print(
                "Epoch completed. Determining hold-out perplexity...", file=sys.stderr
            )
        pp = lm_perplexity(model, val, options.device)
        do_aug |= pp < aug_pp_thresh
        controller.update_for_epoch(
            model, optimizer, train_loss, pp, epoch, do_aug=int(do_aug)
        )
        if options.quiet < 2:
            print(
                f"Epoch {epoch}: Train loss={train_loss:.02f}, hold-out "
                f"perplexity={pp:.02f}",
                file=sys.stderr,
            )
        epoch += 1

    if options.quiet < 2:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if options.model_dir is not None:
        epoch = controller.get_best_epoch()
        if options.quiet < 2:
            print(
                f"Best epoch was {epoch}. Saving that model", file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
    elif options.quiet < 2:
        print(
            f"No history kept. Saving model from last epoch ({epoch - 1})",
            file=sys.stderr,
        )

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(options.model_dir, "final.pt"))


def eval_lm(options, dict_):
    data_dir = os.path.join(options.data_dir, "test", "ref")
    if not os.path.isdir(data_dir):
        raise ValueError(f"'{data_dir}' is not a directory. Did you initialize it?")

    if options.load_path is not None:
        print(f"model: {options.load_path.name}, ", end="")
        model = init_model(options, dict_["model"], dict_["am"]["training"].delta_order)
        if options.full_model:
            model.load_state_dict(torch.load(options.load_path, options.device), False)
        model = model.conditional
        model.add_module("post_merger", None)
        model.add_module("input_merger", None)
        if not options.full_model:
            model.load_state_dict(torch.load(options.load_path, options.device))
        model = model.to(options.device)
    else:
        print("model: arpa.lm.gz, ", end="")
        # FIXME(sdrobert): windows
        arpa_lm_path = glob.glob(f"{options.data_dir}//**/lm.arpa.gz")
        if len(arpa_lm_path) == 0:
            raise ValueError(
                f"Could not find lm.arpa.gz in '{data_dir}'. Did you make it?"
            )
        elif len(arpa_lm_path) > 1:
            raise ValueError(
                f"found multiple lm.arpa.gz files in '{data_dir}': {arpa_lm_path}"
            )
        arpa_lm_path = arpa_lm_path.pop()
        token2id_path = os.path.join(os.path.dirname(arpa_lm_path), "token2id.txt")
        token2id = dict()
        with open(token2id_path) as file_:
            for line in file_:
                token, id_ = line.strip().split()
                id_ = int(id_)
                token2id[token] = id_
        m = max(token2id.values())
        token2id["<s>"] = m + 1
        token2id["</s>"] = m + 2
        with gzip.open(arpa_lm_path, mode="rt") as file_:
            prob_list = data.parse_arpa_lm(file_, token2id)
        model = modules.LookupLanguageModel(len(token2id), token2id["<s>"], prob_list)
        model = model.to(options.device)

    dev = data.LangDataSet(data_dir)
    lm_pp = lm_perplexity(model, dev, options.device)
    print(f"perplexity: {lm_pp:.02f}")


@torch.no_grad()
def val_error_rates(
    model: models.AcousticModel,
    loader: data.SpectEvaluationDataLoader,
    device: torch.device,
    is_ctc: bool,
    quiet: int,
    pretraining: bool,
):
    model.eval()

    total_errs = 0
    total_toks = 0

    if is_ctc:
        search = modules.CTCGreedySearch()
    else:
        search = modules.BeamSearch(model, 1)
    rater = modules.ErrorRate(eos=config.INDEX_PAD_VALUE, norm=False)

    if quiet < 1:
        loader = tqdm(loader)

    for feats, refs, feat_lens, ref_lens in loader:
        # feats = feats[::3]
        # feat_lens = torch.div(feat_lens - 1, 3, rounding_mode="floor") + 1
        total_toks += ref_lens.sum().item()
        feats, refs = feats.to(device), refs.to(device)
        feat_lens, ref_lens = feat_lens.to(device), ref_lens.to(device)
        if pretraining:
            refs = refs.clamp_max_(0)
        T, N = feats.shape[:2]
        Tp = model.compute_output_time_size(T)
        prev = {"input": feats, "length": feat_lens}
        if is_ctc:
            lprobs = model.calc_ctc_log_probs(refs, prev)
            _, hyps, lens = search(lprobs)
            len_mask = torch.arange(hyps.size(0), device=device).unsqueeze(1) >= lens
            hyps.masked_fill_(len_mask, config.INDEX_PAD_VALUE)
        else:
            hyps = search(prev, batch_size=N, max_iters=Tp)[0].squeeze(-1)
            hyps = models.extended_hist_to_conditional(
                hyps, vocab_size=model.vocab_size - 1
            )[0]
        total_errs += rater(refs, hyps).sum().item()

    return total_errs / total_toks


class TrainingAmWrapper(torch.nn.Module):

    am: models.AcousticModel
    estimator_name: str
    centering: bool
    M: int
    B: int
    ec: float
    pretraining: bool
    mse: bool
    pcb: Optional[models.AcousticModel]
    adaptive: bool

    def __init__(
        self,
        am: models.AcousticModel,
        estimator_name: str,
        M: int,
        B: int,
        pcb: models.AcousticModel = None,
        centering: bool = False,
    ):
        super().__init__()
        self.am = am
        self.estimator_name, self.M, self.B = estimator_name, M, B
        self.ec = 0.0
        self.pretraining = self.mse = self.adaptive = False
        self.pcb = pcb
        self.centering = centering

    def forward_regular(
        self,
        feats: torch.Tensor,
        refs: torch.Tensor,
        feat_lens: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        T, N = feats.shape[:2]

        if self.estimator_name == "marginal":
            prev = {
                "input": feats,
                "length": feat_lens,
            }
            v = self.am.calc_marginal_log_likelihoods(refs, ref_lens, prev)
        elif self.estimator_name == "ctc":
            prev = {
                "input": feats,
                "length": feat_lens,
            }
            lens_ = self.am.compute_output_time_size(feat_lens)
            lprobs = self.am.calc_ctc_log_probs(refs, prev)
            v = -torch.nn.functional.ctc_loss(
                lprobs, refs.T, lens_, ref_lens, self.am.vocab_size - 1, "mean"
            )
        else:

            def func(b: torch.Tensor) -> torch.Tensor:
                M = b.size(0)
                mismatch = b.sum(-1) != func.ref_lens  # (M, N)
                h = func.h.unsqueeze(1).expand(-1, M, -1, -1).flatten(1, 2)
                h_lens = func.h_lens.unsqueeze(0).expand(M, -1).flatten()
                refs = func.refs.unsqueeze(1).expand(-1, M, -1).flatten(1)
                b_ = b.flatten(0, 1).T.long()
                ll = self.am.calc_log_likelihood_given_latents(
                    b_, refs, {"latent_input": h, "latent_length": h_lens},
                )
                ll = ll.view(b.shape[:-1]).masked_fill(mismatch, -1e10)
                if func.centering and M > 1:
                    center = (
                        (1.0 - torch.eye(M, device=b.device).unsqueeze(1))
                        * ll.T.detach()
                    ).logsumexp(2) - math.log(M - 1)
                    ll = ll.logaddexp(-center)
                return ll

            Tp = self.am.compute_output_time_size(T).item()
            h, lens_ = self.am.frontend(feats.transpose(0, 1), feat_lens)
            func.h = h
            func.h_lens = lens_
            func.refs = refs
            func.ref_lens = ref_lens
            func.centering = self.centering
            prev = {
                "input": h,
                "post": h,
                "cond_hist": refs,
                "length": lens_,
                "given": ref_lens,
            }
            prev = self.am.latent.update_input(prev, refs)

            if self.pcb is not None and self.estimator_name in {
                "ais-c",
                "ais-g",
                "ais-s",
                "srswor",
            }:
                self.pcb.eval()
                with torch.no_grad():
                    h, lens_ = self.pcb.frontend(feats.transpose(0, 1), feat_lens)
                    prev_ = {"input": h, "length": lens_, "given": ref_lens}
                    hidden = self.pcb.latent.calc_all_hidden(prev_)
                    logits = self.pcb.latent.calc_all_logits(prev_, hidden)
                    del h, lens_, hidden, prev_
                    logits = logits[..., 1] - logits[..., 0]
                    proposal = distributions.ConditionalBernoulli(
                        ref_lens, logits=logits.T
                    )
            elif self.estimator_name in {"ais-c", "ais-g", "ais-s", "srswor"}:
                proposal = pdistributions.SimpleRandomSamplingWithoutReplacement(
                    ref_lens, lens_, Tp
                )
            elif self.estimator_name == "sf-is":
                model_ = models.SuffixForcingWrapper(self.am.latent, "length")
                walk = modules.RandomWalk(model_)
                proposal = pdistributions.SequentialLanguageModelDistribution(
                    walk, N, prev, Tp, True
                )

            v = 0
            if self.estimator_name == "cb":
                hidden = self.am.latent.calc_all_hidden(prev)
                logits = self.am.latent.calc_all_logits(prev, hidden)
                logits = logits[..., 1] - logits[..., 0]
                dist = distributions.ConditionalBernoulli(ref_lens, logits=logits.T)
                v = distributions.PoissonBinomial(logits=logits.T).log_prob(ref_lens)
            else:
                if self.estimator_name == "sf-biased":
                    model_ = models.SuffixForcingWrapper(self.am.latent, "length")
                    walk = modules.RandomWalk(model_)
                else:
                    walk = modules.RandomWalk(self.am.latent)
                dist = pdistributions.SequentialLanguageModelDistribution(
                    walk,
                    N,
                    prev,
                    Tp,
                    # self.estimator_name in {"direct", "cb", "sf-biased", "sf-is"},
                )
            if self.estimator_name in {"direct", "cb", "sf-biased"}:
                estimator = pestimators.DirectEstimator(dist, func, self.M, is_log=True)
            elif not self.adaptive:
                estimator = pestimators.ImportanceSamplingEstimator(
                    proposal, func, self.M, dist, is_log=True,
                )
            else:
                assert self.estimator_name.startswith("ais-")
                maker = estimators.ConditionalBernoulliProposalMaker(ref_lens)
                density = pdistributions.SequentialLanguageModelDistribution(
                    walk, N, prev, Tp
                )
                if self.estimator_name == "ais-c":
                    adaptation_func = lambda b: b
                elif self.estimator_name == "ais-g":
                    adaptation_func = estimators.FixedCardinalityGibbsStatistic(
                        func, density, True
                    )
                elif self.estimator_name == "ais-s":

                    @torch.no_grad()
                    def adaptation_func(b: torch.Tensor) -> torch.Tensor:
                        logits = self.am.latent.calc_full_log_probs(
                            b.T[:-1].long(), prev
                        )
                        return (logits[..., 1] - logits[..., 0]).T.softmax(1)

                else:
                    assert False
                estimator = estimators.AisImhEstimator(
                    proposal,
                    func,
                    self.M,
                    density,
                    adaptation_func,
                    maker,
                    self.B,
                    is_log=True,
                    centering=self.centering,
                )
            v = v + estimator()
        return -v.mean()

    def forward_pretrain(
        self,
        feats: torch.Tensor,
        refs: torch.Tensor,
        feat_lens: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        T, N = feats.shape[:2]

        def func(b: torch.Tensor) -> torch.Tensor:
            out_lens = b.sum(-1)
            return -(out_lens - func.ref_lens).float() ** 2

        Tp = self.am.compute_output_time_size(T).item()
        h, lens_ = self.am.frontend(feats.transpose(0, 1), feat_lens)
        func.ref_lens = ref_lens
        prev = {
            "input": h,
            "post": h,
            "length": lens_,
            "given": ref_lens,
        }
        prev = self.am.latent.update_input(prev, refs)

        if self.estimator_name in {"marginal", "cb"}:
            hidden = self.am.latent.calc_all_hidden(prev)
            logits = self.am.latent.calc_all_logits(prev, hidden)
            logits = logits[..., 1] - logits[..., 0]
            v = distributions.PoissonBinomial(logits=logits.T).log_prob(ref_lens)
        elif self.estimator_name == "ctc":
            hidden = self.am.latent.calc_all_hidden(prev)
            lprobs = self.am.latent.calc_all_logits(prev, hidden).log_softmax(2).flip(2)
            v = -torch.nn.functional.ctc_loss(
                lprobs, refs.T, lens_, ref_lens, 1, "mean"
            )
        else:
            walk = modules.RandomWalk(self.am.latent)
            dist = pdistributions.SequentialLanguageModelDistribution(walk, N, prev, Tp)
            estimator = pestimators.DirectEstimator(dist, func, self.M, is_log=False)
            v = estimator()
        return -v.mean()

    def forward(
        self,
        feats: torch.Tensor,
        refs: torch.Tensor,
        feat_lens: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        if not self.pretraining:
            return self.forward_regular(feats, refs, feat_lens, ref_lens)
        else:
            return self.forward_pretrain(feats, refs, feat_lens, ref_lens)

    def reset_parameters(self):
        self.am.reset_parameters()


def train_am_for_epoch(
    model: TrainingAmWrapper,
    loader: data.SpectTrainingDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    params: AcousticModelTrainingParams,
    epoch: int,
    device: torch.device,
    quiet: int,
    pretraining: bool,
    do_aug: bool,
    world_size: int,
) -> float:
    loader.epoch = epoch
    non_blocking = False  # device.type == "cpu" or loader.pin_memory
    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)
    torch.manual_seed((epoch + 5) * (controller.params.seed + 113))
    sa = lambda x, _: x
    wn_setup = wn_teardown = lambda: None
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_ = model.module
    else:
        model_ = model
    if epoch > 1:
        model_.adaptive = params.estimator.startswith("ais-")
    if do_aug:
        model_.am.dropout_prob = params.dropout_prob
        model_.am.swap_prob = params.swap_prob
        if params.entropy_init:
            model_.ec = (
                params.entropy_init
                * params.entropy_decay ** ((epoch - 1) * len(loader))
                + params.entropy_floor
            )
        if (params.sa_time_num_prop and params.sa_time_size_prop) or (
            params.sa_freq_num and params.sa_freq_size
        ):
            sa = modules.SpecAugment(
                0,
                0,
                100000,
                params.sa_freq_size,
                params.sa_time_size_prop,
                100000,
                params.sa_time_num_prop,
                params.sa_freq_num,
            )
        if params.weight_noise_std:
            param_data = []

            def wn_setup():
                for p in model.parameters():
                    param_data.append(p.data.clone())
                    p.data += (
                        torch.randn_like(p)
                        * params.weight_noise_std
                        / math.sqrt(max(world_size, 1))
                    )

            def wn_teardown():
                for p, p_ in zip(model.parameters(), param_data):
                    # we copy so that RNN parameters remain contiguous in memory
                    p.data.copy_(p_)
                param_data.clear()

    elif params.entropy_init:
        model.ec = params.entropy_init

    model.train()

    num_samples = len(loader.batch_sampler.sampler)

    if quiet < 1:
        loader = tqdm(loader)

    total_loss = 0.0
    for feats, refs, feat_lens, ref_lens in loader:
        feats = feats.to(device, non_blocking=non_blocking)
        refs = refs.to(device, non_blocking=non_blocking)
        if pretraining:
            refs = refs.clamp_max_(0)
        feat_lens = feat_lens.to(device, non_blocking=non_blocking)
        ref_lens = ref_lens.to(device, non_blocking=non_blocking)
        feats = sa(feats.transpose(0, 1), feat_lens).transpose(0, 1)

        optimizer.zero_grad()
        wn_setup()
        loss = model(feats, refs, feat_lens, ref_lens)
        loss.backward()
        total_loss += loss.detach() * feat_lens.size(0)
        wn_teardown()
        optimizer.step()

    return total_loss.item() / num_samples


def train_am(options, dict_):
    os.makedirs(options.model_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if options.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if rank >= 0:
        if rank != 0:
            options.quiet = 100
        if options.device.type == "cuda":
            if local_rank >= 0:
                options.device = torch.device(local_rank)
                torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                "nccl",
                rank=rank,
                world_size=world_size,
                # timeout=datetime.timedelta(0, 10),
            )
        else:
            torch.distributed.init_process_group(
                "gloo",
                rank=rank,
                world_size=world_size,
                # timeout=datetime.timedelta(0, 10),
            )
    if world_size != dict_["am"]["training"].world_size:
        raise ValueError(
            f'Configured world size ({dict_["am"]["training"].world_size}) is not the '
            f"same as actual world size ({world_size})"
        )

    num_data_workers = min(get_num_avail_cores() - 1, 4)
    train_dir = os.path.join(options.data_dir, "train")
    if not os.path.isdir(train_dir):
        raise ValueError(f"'{train_dir}' is not a directory. Did you initialize it?")
    dev_dir = os.path.join(options.data_dir, "dev")
    if not os.path.isdir(dev_dir):
        raise ValueError(f"'{dev_dir}' is not a directory. Did you initialize it?")
    seed = dict_["am"]["training"].seed

    model_ = model = init_model(
        options,
        dict_["model"],
        dict_["am"]["training"].delta_order,
        options.pretraining,
    )
    if dict_["am"]["training"].estimator == "pcb" or options.pcb_path is not None:
        pcb = init_pcb(options, dict_["am"]["pcb"], dict_["am"]["training"].delta_order)
    else:
        pcb = None
    model = TrainingAmWrapper(
        model,
        dict_["am"]["training"].estimator,
        dict_["am"]["training"].mc_samples,
        dict_["am"]["training"].mc_burn_in,
        pcb,
        dict_["am"]["training"].centering,
    )
    model.pretraining = options.pretraining
    if rank >= 0:
        if local_rank >= 0:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                # find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    ps = (p for p in model.parameters() if p.requires_grad)
    if dict_["am"]["training"].optimizer == "sgd":
        optimizer = torch.optim.SGD(ps, 1e-4, momentum=dict_["am"]["training"].momentum)
    elif dict_["am"]["training"].optimizer == "adam":
        optimizer = torch.optim.Adam(ps)
    else:
        assert False

    state_csv = os.path.join(options.model_dir, "hist.csv")
    state_dir = os.path.join(
        options.model_dir if options.ckpt_dir is None else options.ckpt_dir, "training"
    )
    os.makedirs(state_dir, exist_ok=True)

    controller = training.TrainingStateController(
        dict_["am"]["training"], state_csv, state_dir, warn=options.quiet < 1
    )

    stats_file = os.path.join(options.data_dir, "ext", "train.mvn.pt")
    stats = torch.load(stats_file)

    train_loader = data.SpectDataLoader(
        train_dir,
        dict_["am"]["training"],
        shuffle=True,
        seed=seed,
        num_workers=num_data_workers,
        batch_first=False,
        pin_memory=False,
        feat_mean=stats["mean"],
        feat_std=stats["std"],
    )

    dev_loader = data.SpectDataLoader(
        dev_dir,
        dict_["am"]["training"],
        shuffle=False,
        batch_first=False,
        num_workers=num_data_workers,
        feat_mean=stats["mean"],
        feat_std=stats["std"],
    )

    # dev_er = float("inf")
    epoch = controller.get_last_epoch() + 1
    aug_er_thresh = dict_["am"]["training"].aug_er_thresh
    if epoch == 1:
        best_er = 1000
    else:
        best_er = controller.get_info(controller.get_best_epoch())["val_met"]
    do_aug = best_er < aug_er_thresh
    while controller.continue_training(epoch - 1):
        if options.quiet < 1:
            print(f"Training epoch {epoch}...", file=sys.stderr)
        train_loss = train_am_for_epoch(
            model,
            train_loader,
            optimizer,
            controller,
            dict_["am"]["training"],
            epoch,
            options.device,
            options.quiet,
            options.pretraining,
            do_aug,
            options.ddp_world_size,
        )
        if options.quiet < 1:
            print("Epoch completed. Determining dev error rate...", file=sys.stderr)
        dev_er = val_error_rates(
            model_,
            dev_loader,
            options.device,
            dict_["am"]["decoding"].is_ctc,
            options.quiet,
            options.pretraining,
        )

        controller.update_for_epoch(
            model,
            optimizer,
            train_loss,
            dev_er,
            epoch,
            # aug_er_patience_cd=aug_er_patience_cd,
        )
        best_er = min(controller.get_info(epoch)["val_met"], best_er)
        do_aug = best_er < aug_er_thresh
        if options.quiet < 2:
            print(
                f"Epoch {epoch}: Train loss={train_loss:e}, dev error rate={dev_er:%}",
                file=sys.stderr,
            )
        epoch += 1

    if options.quiet < 2:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if options.model_dir is not None:
        epoch = controller.get_best_epoch()
        if options.quiet < 2:
            print(
                f"Best epoch was {epoch}. Saving that model", file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
    elif options.quiet < 2:
        print(
            f"No history kept. Saving model from last epoch ({epoch - 1})",
            file=sys.stderr,
        )

    if rank <= 0:
        state_dict = model_.to("cpu").state_dict()
        if options.pretraining:
            for key in tuple(state_dict.keys()):
                if key.startswith("conditional."):
                    state_dict.pop(key)
        torch.save(state_dict, os.path.join(options.model_dir, "final.pt"))


@torch.no_grad()
def decode_am(options, dict_):
    test_dir = os.path.join(options.data_dir, "dev" if options.dev else "test")

    model = init_model(options, dict_["model"], dict_["am"]["training"].delta_order)
    model.eval()

    if dict_["am"]["decoding"].style == "beam":
        if dict_["am"]["decoding"].is_ctc:
            raise NotImplementedError
        search = modules.BeamSearch(model, options.beam_width)
    elif dict_["am"]["decoding"].is_ctc:
        search = modules.CTCPrefixSearch(options.beam_width)
    else:
        search = decoding.JointLatentLanguageModelPrefixSearch(
            model, options.beam_width
        )

    stats_file = os.path.join(options.data_dir, "ext", "train.mvn.pt")
    stats = torch.load(stats_file)

    test_set = data.SpectDataSet(
        test_dir,
        params=dict_["am"]["training"],
        feat_mean=stats["mean"],
        feat_std=stats["std"],
        suppress_alis=True,
        suppress_uttids=False,
        tokens_only=False,
    )
    for feats, _, utt_id in test_set:
        T = feats.size(0)
        feats = feats.to(options.device).unsqueeze(1)
        feat_lens = torch.tensor([T]).to(options.device)
        prev = {"input": feats, "length": feat_lens}
        Tp = model.compute_output_time_size(T)
        if dict_["am"]["decoding"].style == "beam":
            hyp = search(prev, batch_size=1, max_iters=Tp.item())[0][..., 0]
            hyp, len_ = models.extended_hist_to_conditional(
                hyp, vocab_size=model.vocab_size - 1
            )
        elif dict_["am"]["decoding"].is_ctc:
            lprobs = model.calc_ctc_log_probs(feat_lens.new_empty((0, 1)), prev)
            hyp, len_, _ = search(lprobs, Tp.view(1), prev)
            hyp, len_ = hyp[..., 0], len_[..., 0]
        else:
            hyp, len_, _ = search(1, Tp.item(), prev)
            hyp, len_ = hyp[..., 0], len_[..., 0]
        hyp = hyp.flatten()[: len_.flatten()].cpu()
        torch.save(hyp, f"{options.hyp_dir}/{utt_id}.pt")


class DirType(object):

    mode: str

    def __init__(self, mode: Literal["r", "w"]):
        self.mode = mode

    def __repr__(self) -> str:
        return f"Dir('{self.mode}')"

    def __call__(self, path: str) -> str:
        if self.mode == "r":
            if not os.path.isdir(path):
                raise TypeError(f"path '{path}' is not an existing directory")
        else:
            os.makedirs(path, exist_ok=True)
        return path


class FileType(object):

    mode: str

    def __init__(self, mode: Literal["r", "w"]):
        self.mode = mode

    def __repr__(self) -> str:
        return f"File('{self.mode}')"

    def __call__(self, path: str) -> str:
        if self.mode == "r":
            if not os.path.isfile(path):
                raise TypeError(f"path '{path}' is not an existing file")
        elif not os.path.isdir(os.path.dirname(path)):
            raise TypeError(f"path '{path}' parent folder does not exist")
        return path


@record
def main(args=None):

    parser = argparse.ArgumentParser(description="Commands for running ASR experiments")
    dict_ = construct_default_param_dict()
    pargparse.add_parameterized_read_group(
        parser, parameterized=dict_, ini_option_strings=tuple()
    )
    pargparse.add_parameterized_print_group(
        parser, parameterized=dict_, ini_option_strings=tuple()
    )
    parser.add_argument("data_dir", type=DirType("r"))
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--ckpt-dir", type=DirType("w"), default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="Clobber config seeds with this if set"
    )
    parser.add_argument("--quiet", "-q", action="count", default=0)
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")

    train_lm_parser = subparsers.add_parser("train_lm")
    train_lm_parser.add_argument("model_dir", type=DirType("w"))

    eval_lm_parser = subparsers.add_parser("eval_lm")
    eval_lm_parser.add_argument(
        "load_path", nargs="?", type=argparse.FileType("rb"), default=None
    )
    eval_lm_parser.add_argument("--full-model", action="store_true", default=False)

    train_am_parser = subparsers.add_parser("train_am")
    train_am_parser.add_argument("model_dir", type=DirType("w"))
    train_am_parser.add_argument(
        "--pretrained-enc-path", type=FileType("r"), default=None
    )
    train_am_parser.add_argument("--pcb-path", type=FileType("r"), default=None)
    train_am_parser.add_argument("--ddp-world-size", default=0, type=int)
    encoder_group = train_am_parser.add_mutually_exclusive_group()
    encoder_group.add_argument("--pretraining", action="store_true", default=False)
    encoder_group.add_argument("--pretrained-lm-path", type=FileType("r"), default=None)

    decode_am_parser = subparsers.add_parser("decode_am")
    decode_am_parser.add_argument("am_path", type=argparse.FileType("rb"))
    decode_am_parser.add_argument("hyp_dir", type=DirType("w"))
    decode_am_parser.add_argument("--dev", action="store_true", default=False)
    decode_am_parser.add_argument("--beam-width", type=int, default=1)

    options = parser.parse_args(args)

    if options.seed is not None:
        dict_["am"]["training"].seed = dict_["lm"].seed = options.seed

    if options.command == "train_lm":
        return train_lm(options, dict_)
    elif options.command == "eval_lm":
        return eval_lm(options, dict_)
    elif options.command == "train_am":
        return train_am(options, dict_)
    elif options.command == "decode_am":
        return decode_am(options, dict_)


if __name__ == "__main__":
    sys.exit(main())
