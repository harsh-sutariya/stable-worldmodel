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


def build_wandb_logger(cfg, run_dir: Path | None = None):
    """Return a WandbLogger if wandb.enabled is true, otherwise None.

    If run_dir is provided, the W&B run ID is persisted to
    <run_dir>/wandb_run_id.txt so that resuming training always
    continues the exact same W&B run (rather than starting a new one).
    Delete that file to force a fresh W&B run.
    """
    import wandb as _wandb
    from lightning.pytorch.loggers import WandbLogger

    if not cfg.wandb.enabled:
        return None

    kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    kwargs.pop('enabled')
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if run_dir is not None:
        id_file = run_dir / 'wandb_run_id.txt'
        if id_file.exists():
            saved_id = id_file.read_text().strip()
            kwargs['id'] = saved_id
            kwargs['resume'] = 'must'
            logging.info(f'Resuming W&B run {saved_id}')
        else:
            new_id = _wandb.util.generate_id()
            id_file.write_text(new_id)
            kwargs['id'] = new_id
            kwargs.pop('resume', None)
            logging.info(f'Starting new W&B run {new_id}')

    pl_logger = WandbLogger(**kwargs)
    pl_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return pl_logger


class SaveCkptCallback(Callback):
    """Save deployable .pt weights every N epochs via save_pretrained.

    Distinct from Lightning's ModelCheckpoint (which stores full trainer
    state for resumption). This callback writes the slim weights that
    load_pretrained / AutoCostModel / AutoActionableModel can read.

    If probe_cfg is provided, linear probing is run after each checkpoint
    save and R² curves are logged to W&B (same global_step x-axis as loss).
    """

    def __init__(
        self,
        run_name: str,
        cfg,
        every_n_epochs: int = 5,
        probe_cfg: dict | None = None,
        device: str = 'cpu',
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.every_n_epochs = every_n_epochs
        self.probe_cfg = probe_cfg
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0 or epoch == trainer.max_epochs:
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                config_key='model',
                filename=f'weights_epoch_{epoch:04d}.pt',
            )
            if self.probe_cfg is not None:
                self._run_probe(pl_module.model, trainer.global_step)

    def _run_probe(self, model, global_step: int):
        import wandb
        from probe import probe_model

        logging.info(f'Running linear probe at step {global_step}...')
        results = probe_model(model, self.probe_cfg, self.device)

        metrics = {}
        for level, targets in results.items():
            for target, v in targets.items():
                metrics[f'probe/{level}/{target}'] = v['mean']

        if wandb.run is not None:
            wandb.log(metrics, step=global_step)
            logging.info(f'Probe metrics logged to W&B ({len(metrics)} keys)')
