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
from torch.utils.data import DataLoader

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss

from utils import SaveCkptCallback, build_wandb_logger, get_img_preprocessor, setup_run_dir


def pldm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)
    emb = output['emb']       # (B, T, D)
    act_emb = output['act_emb']

    inpt_emb = emb[:, : cfg.wm.history_size]
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds:]
    pred_emb = self.model.predict(inpt_emb, inpt_act)

    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - tgt_emb).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))

    output['loss'] = output['pred_loss']
    for k, v in cfg.loss.items():
        loss_key = f'{k}_loss'
        if not v.enabled or (loss_key not in output):
            continue
        output['loss'] = output['loss'] + v.weight * output[loss_key]

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path='./config', config_name='pldm')
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

    img_processor = get_img_preprocessor('pixels', 'pixels', cfg.img_size)
    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col == 'pixels':
            continue
        extra_transforms.append(get_column_normalizer(dataset, col, col))

    if hasattr(cfg.data.dataset, 'keys_to_merge'):
        for col in cfg.data.dataset.keys_to_merge:
            extra_transforms.append(get_column_normalizer(dataset, col, col))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col == 'pixels':
                continue
            setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))

        effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
        cfg.model.action_encoder.input_dim = effective_act_dim
        cfg.idm.input_dim = 2 * cfg.wm.embed_dim
        cfg.idm.output_dim = effective_act_dim

    dataset.transform = spt.data.transforms.Compose(img_processor, *extra_transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    idm = hydra.utils.instantiate(cfg.idm)
    models = {'model': world_model, 'idm': idm}
    losses = {'pldm': PLDMLoss(), 'path_straight': TemporalStraighteningLoss()}

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        f'{name}_opt': {
            'modules': name,
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }
        for name in models
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models,
        **losses,
        forward=partial(pldm_forward, cfg=cfg),
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
