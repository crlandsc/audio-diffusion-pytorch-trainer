# Code modified from audio-diffusion-pytorch package from Archinet (https://github.com/archinetai/audio-diffusion-pytorch)

from functools import reduce
from inspect import isfunction
from math import ceil, floor, log2, pi
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Generator, Tensor
from typing_extensions import TypeGuard

import logging
import os
import warnings

import pkg_resources  # type: ignore
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


"""
Kwargs Utils
"""


def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}


"""
DSP Utils
"""


def resample(
    waveforms: Tensor,
    factor_in: int,
    factor_out: int,
    rolloff: float = 0.99,
    lowpass_filter_width: int = 6,
) -> Tensor:
    """Resamples a waveform using sinc interpolation, adapted from torchaudio"""
    b, _, length = waveforms.shape
    length_target = int(factor_out * length / factor_in)
    d = dict(device=waveforms.device, dtype=waveforms.dtype)

    base_factor = min(factor_in, factor_out) * rolloff
    width = ceil(lowpass_filter_width * factor_in / base_factor)
    idx = torch.arange(-width, width + factor_in, **d)[None, None] / factor_in  # type: ignore # noqa
    t = torch.arange(0, -factor_out, step=-1, **d)[:, None, None] / factor_out + idx  # type: ignore # noqa
    t = (t * base_factor).clamp(-lowpass_filter_width, lowpass_filter_width) * pi

    window = torch.cos(t / lowpass_filter_width / 2) ** 2
    scale = base_factor / factor_in
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    waveforms = rearrange(waveforms, "b c t -> (b c) t")
    waveforms = F.pad(waveforms, (width, width + factor_in))
    resampled = F.conv1d(waveforms[:, None], kernels, stride=factor_in)
    resampled = rearrange(resampled, "(b c) k l -> b c (l k)", b=b)
    return resampled[..., :length_target]


def downsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    return resample(waveforms, factor_in=factor, factor_out=1, **kwargs)


def upsample(waveforms: Tensor, factor: int, **kwargs) -> Tensor:
    return resample(waveforms, factor_in=1, factor_out=factor, **kwargs)


""" Torch Utils """


def randn_like(tensor: Tensor, *args, generator: Optional[Generator] = None, **kwargs):
    """randn_like that supports generator"""
    return torch.randn(tensor.shape, *args, generator=generator, **kwargs).to(tensor)



""" Training Utils"""

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    hparams["pacakges"] = get_packages_list()

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def get_packages_list() -> List[str]:
    return [f"{p.project_name}=={p.version}" for p in pkg_resources.working_set]


def retry_if_error(fn: Callable, num_attemps: int = 10):
    for attempt in range(num_attemps):
        try:
            return fn()
        except:
            print(f"Retrying, attempt {attempt+1}")
            pass
    return fn()


class SavePytorchModelAndStopCallback(Callback):
    def __init__(self, path: str, attribute: Optional[str] = None):
        self.path = path
        self.attribute = attribute

    def on_train_start(self, trainer, pl_module):
        model, path = pl_module, self.path
        if self.attribute is not None:
            assert_message = "provided model attribute not found in pl_module"
            assert hasattr(pl_module, self.attribute), assert_message
            model = getattr(
                pl_module, self.attribute, hasattr(pl_module, self.attribute)
            )
        # Make dir if not existent
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        # Save model
        torch.save(model, path)
        log.info(f"PyTorch model saved at: {path}")
        # Stop trainer
        trainer.should_stop = True
