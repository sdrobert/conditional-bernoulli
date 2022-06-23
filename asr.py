import sys
import argparse
import os
import random
import math
import glob
import gzip

from typing import Optional, Set, Tuple
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

import models
import distributions
import estimators


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


class AcousticModelTrainingStateParams(training.TrainingStateParams):
    estimator = param.ObjectSelector(
        "direct", objects=["direct", "marginal", "partial-indep", "srswor"]
    )
    mc_samples = param.Integer(1, bounds=(1, None))
    mc_burn_in = param.Integer(1, bounds=(1, None))
    dropout_prob = param.Magnitude(0.0)


class LanguageModelTrainingStateParams(training.TrainingStateParams):
    fraction_held_out = param.Magnitude(0.05)
    dropout_prob = param.Magnitude(0.0)
    swap_prob = param.Magnitude(0.0)


class LanguageModelDataLoaderParams(param.Parameterized):
    batch_size = param.Integer(10, bounds=(1, None))
    drop_last = param.Boolean(False)


class DecodingParams(param.Parameterized):
    beam_width = param.Integer(4, bounds=(1, None))


class LanguageModelDataSet(torch.utils.data.Dataset):

    data_dir: str
    utt_ids: Tuple[str, ...]

    def __init__(self, data_dir: str, subset_ids: Optional[Set[str]] = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.subset_ids = subset_ids
        utt_ids = self.get_utt_ids_in_data_dir(data_dir)
        if subset_ids is not None:
            assert subset_ids & utt_ids == subset_ids
            utt_ids = subset_ids
        self.utt_ids = sorted(utt_ids)

    @staticmethod
    def get_utt_ids_in_data_dir(data_dir: str) -> Set[str]:
        assert os.path.isdir(data_dir)
        return {u[:-3] for u in os.listdir(data_dir) if u.endswith(".pt")}

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.data_dir, self.utt_ids[idx] + ".pt")
        return torch.load(path)[..., 0]


class LanguageModelDataLoader(torch.utils.data.DataLoader):

    params: LanguageModelDataLoaderParams
    init_epoch: int

    def __init__(
        self,
        data_dir: str,
        params: LanguageModelDataLoaderParams,
        init_epoch: int = 0,
        subset_ids: Optional[Set[str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        for bad_kwarg in (
            "batch_size",
            "sampler",
            "batch_sampler",
            "shuffle",
            "collate_fn",
        ):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)
                    )
                )
        self.params = params
        data_source = LanguageModelDataSet(data_dir, subset_ids)
        epoch_sampler = data.EpochRandomSampler(data_source, init_epoch, seed)
        batch_sampler = torch.utils.data.BatchSampler(
            epoch_sampler, params.batch_size, drop_last=params.drop_last
        )
        super().__init__(
            data_source,
            batch_sampler=batch_sampler,
            collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(
                x, padding_value=config.INDEX_PAD_VALUE
            ),
            **kwargs,
        )

    @property
    def epoch(self) -> int:
        return self.batch_sampler.sampler.epoch

    @epoch.setter
    def epoch(self, val: int):
        self.batch_sampler.sampler.epoch = val


def construct_default_param_dict():
    return {
        "model": {
            "latent": models.LstmLmParams(name="model.latent"),
            "conditional": models.LstmLmParams(name="model.conditional"),
            "merge": MergeParams(name="model.merge"),
        },
        "am": {
            "data": {
                "train": data.SpectDataLoaderParams(name="am.data.train"),
                "dev": data.SpectDataLoaderParams(name="am.data.dev"),
                "test": data.SpectDataLoaderParams(name="am.data.test"),
            },
            "training": AcousticModelTrainingStateParams(name="am.training"),
            "decoding": DecodingParams(name="am.decoding"),
        },
        "lm": {
            "data": LanguageModelDataLoaderParams(name="lm.data"),
            "training": LanguageModelTrainingStateParams(name="lm.training"),
        },
    }


class DirType(object):

    mode: str

    def __init__(self, mode: Literal["r", "w"]):
        super().__init__()
        self.mode = mode

    def __call__(self, path: str) -> str:
        if self.mode == "r":
            if not os.path.isdir(path):
                raise TypeError(f"path '{path}' is not an existing directory")
        else:
            os.makedirs(path, exist_ok=True)
        return path


def train_lm_for_epoch(
    model: models.SequentialLanguageModel,
    loader: LanguageModelDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    epoch: int,
    device: torch.device,
    swap_prob: float,
    quiet: bool,
) -> float:
    loader.epoch = epoch
    non_blocking = False  # device.type == "cpu" or loader.pin_memory
    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.INDEX_PAD_VALUE)

    model.train()

    if not quiet:
        loader = tqdm(loader)

    total_loss = 0
    for hyp in loader:
        hyp = hyp.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        hist = hyp[:-1].clamp(0, model.vocab_size - 1)
        if swap_prob != 0:
            mask = torch.rand(hist.shape, device=device) > swap_prob
            hist = mask * hist + ~mask * torch.randint_like(hist, model.vocab_size)
        logits = model(hist)
        assert logits.shape[:-1] == hyp.shape
        loss = loss_fn(logits.flatten(0, 1), hyp.flatten())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def lm_perplexity(
    model: models.SequentialLanguageModel,
    val: LanguageModelDataSet,
    device: torch.device,
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


def initialize_model(options, dict_) -> models.JointLatentLstmLm:
    device = options.device
    num_filts, num_classes = get_filts_and_classes(
        os.path.join(options.data_dir, "train")
    )
    model = models.JointLatentLstmLm(
        num_classes,
        num_filts,
        dict_["latent"],
        dict_["conditional"],
        dict_["merge"].cond_input_is_post,
    )
    if "pretrained_lm_path" in options and options.pretrained_lm_path is not None:
        state_dict = torch.load(options.pretrained_lm_path)
        if "post_merger.weight" in state_dict:
            del state_dict["post_merger.weight"]
        if "post_merger.bias" in state_dict:
            del state_dict["post_merger.bias"]
        model.conditional.load_state_dict(state_dict, strict=False)
        # don't let the controller reset the parameters on the first epoch!
        model.conditional.reset_parameters = lambda *args, **kwargs: None
        if model.conditional.lstm is not None:
            # double checking that loading the state dict does not decouple
            # cell weights from lstm weights
            assert (
                model.conditional.lstm.weight_ih_l0
                is model.conditional.cells[0].weight_ih
            )
    return model.to(device)


def train_lm(options, dict_):
    data_dir = os.path.join(options.data_dir, "lm")
    if not os.path.isdir(data_dir):
        raise ValueError(f"'{data_dir}' is not a directory. Did you initialize it?")
    seed = dict_["lm"]["training"].seed

    if dict_["model"]["conditional"].merge_method == "cat":
        raise NotImplementedError(
            "merge_method == 'cat' LM pretraining not implemented"
        )

    model = initialize_model(options, dict_["model"])
    model = model.conditional
    model.dropout_prob = dict_["lm"]["training"].dropout_prob
    optimizer = torch.optim.Adam(model.parameters())

    if options.model_dir is not None:
        state_dir = os.path.join(options.model_dir, "training")
        state_csv = os.path.join(options.model_dir, "hist.csv")
    else:
        state_dir = state_csv = None

    controller = training.TrainingStateController(
        dict_["lm"]["training"], state_csv, state_dir, warn=not options.quiet
    )

    utt_ids = sorted(LanguageModelDataSet.get_utt_ids_in_data_dir(data_dir))
    fraction_held_out = dict_["lm"]["training"].fraction_held_out
    num_val = max(1, int(len(utt_ids) * fraction_held_out))
    num_train = len(utt_ids) - num_val
    if not options.quiet:
        print(
            f"hold-out is {fraction_held_out:.0%} ({num_val}/{num_train})",
            file=sys.stderr,
        )
    random.Random(seed).shuffle(utt_ids)
    train_utt_ids, val_utt_ids = set(utt_ids[:-num_val]), set(utt_ids[-num_val:])
    assert train_utt_ids and val_utt_ids

    loader = LanguageModelDataLoader(
        data_dir,
        dict_["lm"]["data"],
        0,
        train_utt_ids,
        seed,
        num_workers=min(get_num_avail_cores() - 1, 4),
    )

    val = LanguageModelDataSet(data_dir, val_utt_ids)

    pp = float("inf")
    epoch = controller.get_last_epoch() + 1

    while controller.continue_training(epoch - 1):
        if not options.quiet:
            print(f"Training epoch {epoch}...", file=sys.stderr)
        train_loss = train_lm_for_epoch(
            model,
            loader,
            optimizer,
            controller,
            epoch,
            options.device,
            dict_["lm"]["training"].swap_prob,
            options.quiet,
        )
        if not options.quiet:
            print(
                "Epoch completed. Determining hold-out perplexity...", file=sys.stderr
            )
        pp = lm_perplexity(model, val, options.device)
        controller.update_for_epoch(model, optimizer, train_loss, pp, epoch)
        if not options.quiet:
            print(
                f"Train loss: {train_loss:.02f}, hold-out perplexity: {pp:.02f}",
                file=sys.stderr,
            )
        epoch += 1

    if not options.quiet:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if options.model_dir is not None:
        epoch = controller.get_best_epoch()
        if not options.quiet:
            print(
                f"Best epoch was {epoch}. Saving that model", file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
    elif not options.quiet:
        print(
            f"No history kept. Saving model from last epoch ({epoch - 1})",
            file=sys.stderr,
        )

    state_dict = model.state_dict()
    torch.save(state_dict, options.save_path)


def eval_lm(options, dict_):
    data_dir = os.path.join(options.data_dir, "dev", "ref")
    if not os.path.isdir(data_dir):
        raise ValueError(f"'{data_dir}' is not a directory. Did you initialize it?")

    if options.load_path is not None:
        print(f"model: {options.load_path.name}, ", end="")
        if dict_["model"]["conditional"].merge_method == "cat":
            raise NotImplementedError(
                "merge_method == 'cat' LM pretraining not implemented"
            )

        model = initialize_model(options, dict_["model"])
        model = model.conditional.to(options.device)
        model.load_state_dict(torch.load(options.load_path, options.device))
    else:
        print("model: arpa.lm.gz, ", end="")
        # FIXME(sdrobert): windows
        arpa_lm_path = glob.glob(f"{options.data_dir}/local/**/lm.arpa.gz")
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
        with gzip.open(arpa_lm_path, mode="rt") as file_:
            prob_list = data.parse_arpa_lm(file_, token2id)
        model = modules.LookupLanguageModel(len(token2id), token2id["<s>"], prob_list)
        model = model.to(options.device)

    dev = LanguageModelDataSet(data_dir)
    lm_pp = lm_perplexity(model, dev, options.device)
    print(f"perplexity: {lm_pp:.02f}")


@torch.no_grad()
def error_rates(
    model: models.JointLatentLstmLm,
    loader: data.SpectEvaluationDataLoader,
    width: int,
    device: torch.device,
):
    model.eval()

    total_errs = 0
    total_toks = 0
    search = modules.BeamSearch(model, width)
    rater = modules.ErrorRate(eos=config.INDEX_PAD_VALUE, norm=False)
    for feats, _, refs, feat_lens, ref_lens, _ in loader:
        # feats = feats[::3]
        # feat_lens = torch.div(feat_lens - 1, 3, rounding_mode="floor") + 1
        total_toks += ref_lens.sum().item()
        feats, refs = feats.to(device), refs[..., 0].to(device)
        feat_lens, ref_lens = feat_lens.to(device), ref_lens.to(device)
        T, N = feats.shape[:2]
        prev = {"latent_input": feats, "latent_length": feat_lens}
        hyps = search(prev, batch_size=N, max_iters=T)[0][..., 0]
        hyps = models.extended_hist_to_conditional(
            hyps, vocab_size=model.vocab_size - 1
        )[0]
        total_errs += rater(refs, hyps).sum().item()

    return total_errs / total_toks


def train_am_for_epoch(
    model: models.JointLatentLstmLm,
    loader: data.SpectTrainingDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    params: AcousticModelTrainingStateParams,
    epoch: int,
    device: torch.device,
    quiet: bool,
) -> float:
    loader.epoch = epoch
    non_blocking = device.type == "cpu" or loader.pin_memory
    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)
    estimator_name = params.estimator

    model.train()

    if not quiet:
        loader = tqdm(loader)

    def func(b: torch.Tensor) -> torch.Tensor:
        M = b.size(0)
        mismatch = b.sum(-1) != func.ref_lens  # (M, N)
        feats = func.feats.unsqueeze(1).expand(-1, M, -1, -1).flatten(1, 2)
        feat_lens = func.feat_lens.unsqueeze(0).expand(M, -1).flatten()
        refs = func.refs.unsqueeze(1).expand(-1, M, -1).flatten(1)
        ll = model.calc_log_likelihood_given_latents(
            b.flatten(0, 1).T.long(),
            refs,
            {"latent_input": feats, "latent_lengths": feat_lens},
        )
        ll = ll.view(b.shape[:-1]).masked_fill(mismatch, -10000)
        return ll

    total_loss = 0.0
    for feats, _, refs, feat_lens, ref_lens in loader:
        # feats = feats[::3]
        # feat_lens = torch.div(feat_lens - 1, 3, rounding_mode="floor") + 1
        T, N = feats.shape[:2]
        feats = feats.to(device, non_blocking=non_blocking)
        refs = refs[..., 0].to(device, non_blocking=non_blocking)
        feat_lens = feat_lens.to(device, non_blocking=non_blocking)
        ref_lens = ref_lens.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()

        if estimator_name != "marginal":
            func.feats = feats
            func.feat_lens = feat_lens
            func.refs = refs
            func.ref_lens = ref_lens
            prev = {
                "input": feats,
                "post": feats,
                "length": feat_lens,
            }
            if estimator_name == "partial-indep":
                logits = model.latent.calc_all_logits(
                    prev, model.latent.calc_all_hidden(prev)
                )
                logits = logits[..., 1] - logits[..., 0]
                dist = distributions.ConditionalBernoulli(ref_lens, logits=logits.T)
            else:

                walk = modules.RandomWalk(model.latent)
                dist = pdistributions.SequentialLanguageModelDistribution(
                    walk, N, prev, T, True
                )
            if estimator_name in {"direct", "partial-indep"}:
                estimator = pestimators.DirectEstimator(
                    dist, func, params.mc_samples, is_log=True
                )
            elif estimator_name == "srswor":
                proposal = pdistributions.SimpleRandomSamplingWithoutReplacement(
                    ref_lens, feat_lens, T
                )
                estimator = pestimators.ImportanceSamplingEstimator(
                    proposal, func, params.mc_samples, dist, is_log=True
                )
            else:
                assert False
            v = estimator()
        else:
            prev = {
                "latent_input": feats,
                "latent_post": feats,
                "latent_length": feat_lens,
            }
            v = model.calc_marginal_log_likelihoods(refs, ref_lens, prev)

        loss = -v.mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss


def train_am(options, dict_):
    train_dir = os.path.join(options.data_dir, "train")
    if not os.path.isdir(train_dir):
        raise ValueError(f"'{train_dir}' is not a directory. Did you initialize it?")
    dev_dir = os.path.join(options.data_dir, "dev")
    if not os.path.isdir(dev_dir):
        raise ValueError(f"'{dev_dir}' is not a directory. Did you initialize it?")
    seed = dict_["am"]["training"].seed
    num_data_workers = min(get_num_avail_cores() - 1, 4)

    model = initialize_model(options, dict_["model"])
    model.latent.dropout_prob = dict_["am"]["training"].dropout_prob
    model.conditional.dropout_prob = dict_["am"]["training"].dropout_prob
    optimizer = torch.optim.Adam(model.parameters())

    if options.model_dir is not None:
        state_dir = os.path.join(options.model_dir, "training")
        state_csv = os.path.join(options.model_dir, "hist.csv")
    else:
        state_dir = state_csv = None

    controller = training.TrainingStateController(
        dict_["am"]["training"], state_csv, state_dir, warn=not options.quiet
    )

    train_loader = data.SpectTrainingDataLoader(
        train_dir,
        dict_["am"]["data"]["train"],
        batch_first=False,
        pin_memory=True,
        seed=seed,
        num_workers=num_data_workers,
    )
    dev_loader = data.SpectEvaluationDataLoader(
        dev_dir,
        dict_["am"]["data"]["dev"],
        batch_first=False,
        num_workers=num_data_workers,
    )

    dev_er = float("inf")
    epoch = controller.get_last_epoch() + 1

    while controller.continue_training(epoch - 1):
        if not options.quiet:
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
        )
        if not options.quiet:
            print("Epoch completed. Deteriming dev error rate...", file=sys.stderr)
        dev_er = error_rates(
            model, dev_loader, dict_["am"]["decoding"].beam_width, options.device
        )
        controller.update_for_epoch(model, optimizer, train_loss, dev_er, epoch)
        if not options.quiet:
            print(
                f"Train loss: {train_loss:e}, dev error rate: {dev_er:%}",
                file=sys.stderr,
            )
        epoch += 1

    if not options.quiet:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if options.model_dir is not None:
        epoch = controller.get_best_epoch()
        if not options.quiet:
            print(
                f"Best epoch was {epoch}. Saving that model", file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
    elif not options.quiet:
        print(
            f"No history kept. Saving model from last epoch ({epoch - 1})",
            file=sys.stderr,
        )

    state_dict = model.state_dict()
    torch.save(state_dict, options.save_path)


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
    parser.add_argument("--model-dir", type=DirType("w"), default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="Clobber config seeds with this if set"
    )
    parser.add_argument("--quiet", action="store_true", default=False)
    subparsers = parser.add_subparsers(title="commands", required=True, dest="command")

    train_lm_parser = subparsers.add_parser("train_lm")
    train_lm_parser.add_argument("save_path", type=argparse.FileType("wb"))

    eval_lm_parser = subparsers.add_parser("eval_lm")
    eval_lm_parser.add_argument(
        "load_path", nargs="?", type=argparse.FileType("rb"), default=None
    )

    train_am_parser = subparsers.add_parser("train_am")
    train_am_parser.add_argument("save_path", type=argparse.FileType("wb"))
    train_am_parser.add_argument(
        "--pretrained-lm-path", type=argparse.FileType("rb"), default=None
    )

    options = parser.parse_args(args)

    if options.seed is not None:
        dict_["am"]["training"].seed = dict_["lm"]["training"].seed = options.seed

    if options.command == "train_lm":
        return train_lm(options, dict_)
    elif options.command == "eval_lm":
        return eval_lm(options, dict_)
    elif options.command == "train_am":
        return train_am(options, dict_)


if __name__ == "__main__":
    sys.exit(main())
