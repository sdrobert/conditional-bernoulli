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
import pydrobert.torch.data as data
import pydrobert.torch.training as training
import pydrobert.torch.modules as modules
import pydrobert.param.argparse as pargparse

from tqdm import tqdm

import models


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
    estimator = param.ObjectSelector("direct", objects=["direct"])


class LanguageModelTrainingStateParams(training.TrainingStateParams):
    fraction_eval = param.Magnitude(0.05)


class LanguageModelDataLoaderParams(param.Parameterized):
    batch_size = data.DataLoaderParams.batch_size
    drop_last = data.DataLoaderParams.drop_last


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
            "data": data.DataLoaderParams(name="am.data"),
            "training": AcousticModelTrainingStateParams(name="am.training"),
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
    quiet: bool,
) -> float:
    loader.epoch = epoch
    non_blocking = device.type == "cpu" or loader.pin_memory
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
        logits = model(hyp[:-1].clamp(0, model.vocab_size - 1))
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
    num_val = max(1, int(len(utt_ids) * dict_["lm"]["training"].fraction_eval))
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

    val_p = float("inf")
    epoch = controller.get_last_epoch() + 1

    while controller.continue_training(epoch - 1):
        if not options.quiet:
            print(f"Training epoch {epoch}...", file=sys.stderr)
        train_loss = train_lm_for_epoch(
            model, loader, optimizer, controller, epoch, options.device, options.quiet
        )
        if not options.quiet:
            print(
                "Epoch completed. Determining validation perplexity...", file=sys.stderr
            )
        val_p = lm_perplexity(model, val, options.device)
        controller.update_for_epoch(model, optimizer, train_loss, -val_p, epoch)
        if not options.quiet:
            print(
                f"Train loss: {train_loss:e}, val perplexity: {val_p:e}",
                file=sys.stderr,
            )
        epoch += 1

    if not options.quiet:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if options.model_dir is not None:
        epoch = controller.get_best_epoch()
        if not options.quiet:
            print(
                f"Best epoch was {epoch}. Returning that model (and that perplexity)",
                file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
        val_p = -controller.get_info(epoch)["val_met"]
    elif not options.quiet:
        print(
            f"No history kept. Returning model from last epoch ({epoch - 1})",
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
    print(f"perplexity: {lm_pp:e}")


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

    options = parser.parse_args(args)

    if options.seed is not None:
        dict_["am"]["training"].seed = dict_["lm"]["training"].seed = options.seed

    if options.command == "train_lm":
        return train_lm(options, dict_)
    elif options.command == "eval_lm":
        return eval_lm(options, dict_)


if __name__ == "__main__":
    sys.exit(main())
