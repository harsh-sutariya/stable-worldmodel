"""Shared utilities for stable-worldmodel training scripts."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback
from loguru import logger as logging
from omegaconf import OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """ImageNet-normalised resize pipeline for pixel observations."""
    import stable_pretraining as spt

    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


def setup_run_dir(cfg) -> Path:
    """Resolve and create the run directory, then write config.yaml into it."""
    sub = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), sub)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info(f'Run dir: {run_dir}')
    return run_dir


def build_wandb_logger(cfg):
    """Return a WandbLogger if wandb.enabled is true, otherwise None."""
    from lightning.pytorch.loggers import WandbLogger

    if not cfg.wandb.enabled:
        return None

    kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    kwargs.pop('enabled')
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    pl_logger = WandbLogger(**kwargs)
    pl_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return pl_logger


class SaveCkptCallback(Callback):
    """Save deployable .pt weights every N epochs via save_pretrained.

    Distinct from Lightning's ModelCheckpoint (which stores full trainer
    state for resumption). This callback writes the slim weights that
    load_pretrained / AutoCostModel / AutoActionableModel can read.
    """

    def __init__(self, run_name: str, cfg, every_n_epochs: int = 5):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0 or epoch == trainer.max_epochs:
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f'weights_epoch_{epoch:04d}.pt',
            )
