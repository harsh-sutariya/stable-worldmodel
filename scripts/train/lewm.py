import os
from functools import partial

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg

from utils import SaveCkptCallback, build_wandb_logger, get_img_preprocessor, setup_run_dir


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)
    emb = output['emb']       # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    logging.info(
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [get_img_preprocessor('pixels', 'pixels', cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = setup_run_dir(cfg)
    pl_logger = build_wandb_logger(cfg)

    last_ckpt = run_dir / 'lightning' / 'last.ckpt'

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / 'lightning',
            filename='epoch={epoch:04d}',
            monitor='validate/loss',
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=cfg.checkpointing.save_last,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg,
            every_n_epochs=cfg.checkpointing.every_n_epochs,
        ),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=pl_logger,
    )

    spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=last_ckpt if last_ckpt.exists() else None,
    )()


if __name__ == '__main__':
    run()
