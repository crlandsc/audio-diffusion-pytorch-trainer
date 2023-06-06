import os

import dotenv
import hydra
import torch
import pytorch_lightning as pl
from main import utils
from omegaconf import DictConfig, open_dict

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)
log = utils.get_logger(__name__)

# from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from main import DiffusionModel, UNetV0, VDiffusion, VSampler, module_base

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:

    # Logs config tree
    utils.extras(config)

    # Apply seed for reproducibility
    pl.seed_everything(config.seed)

    # Initialize datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
    datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")

    # Initialize model
    log.info(f"Instantiating model <{config.model._target_}>.")

    # model2 = DiffusionModel(
    #     net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    #     in_channels=2, # U-Net: number of input/output (audio) channels
    #     channels=[8, 32, 64, 128, 256, 512], # U-Net: channels at each layer
    #     factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    #     items=[1, 2, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
    #     attentions=[0, 0, 0, 0, 0, 0], # U-Net: attention enabled/disabled at each layer
    #     attention_heads=8, # U-Net: number of attention heads per attention item
    #     attention_features=64, # U-Net: number of attention features per attention item
    #     diffusion_t=VDiffusion, # The diffusion method used
    #     sampler_t=VSampler, # The diffusion sampler used
    # )

    # model2 = module_base.Model(
    #     lr=1e-4,
    #     lr_beta1=0.95,
    #     lr_beta2=0.999,
    #     lr_eps=1e-6,
    #     lr_weight_decay=1e-3,
    #     ema_beta=0.995,
    #     ema_power=0.7,
    #     model=model2
    # )

    model = hydra.utils.instantiate(config.model, _convert_="partial")

    # Initialize all callbacks (e.g. checkpoints, early stopping)
    callbacks = []

    # If save is provided add callback that saves and stops, to be used with +ckpt
    if "save" in config:
        # Ignore loggers and other callbacks
        with open_dict(config):
            config.pop("loggers")
            config.pop("callbacks")
            config.trainer.num_sanity_val_steps = 0
        attribute, path = config.get("save"), config.get("ckpt_dir")
        filename = os.path.join(path, f"{attribute}.pt")
        callbacks += [utils.SavePytorchModelAndStopCallback(filename, attribute)]

    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>.")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    # Initialize loggers (e.g. wandb)
    loggers = []
    if "loggers" in config:
        for _, lg_conf in config["loggers"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>.")
                # Sometimes wandb throws error if slow connection...
                logger = utils.retry_if_error(
                    lambda: hydra.utils.instantiate(lg_conf, _convert_="partial")
                )
                loggers.append(logger)

    # Initialize trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train with checkpoint if present, otherwise from start
    if "ckpt" in config:
        ckpt = config.get("ckpt")
        log.info(f"Starting training from {ckpt}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

        # # Alternative model load method
        # # Use if loading from checkpoint with pl trainer causes GPU memory spike (CUDA out of memory).
        # checkpoint = torch.load(config.get("ckpt"), map_location='cpu')['state_dict']
        # model.load_state_dict(checkpoint)
        # trainer.fit(model=model, datamodule=datamodule)
    else:
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Print path to best checkpoint
    if (
        not config.trainer.get("fast_dev_run")
        and config.get("train")
        and not config.get("save")
    ):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
