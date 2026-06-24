import os
import sys
from functools import partial
from pathlib import Path

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

# Make scripts/plan importable for eval_model
sys.path.insert(0, str(Path(__file__).parent.parent / 'plan'))


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

    pt = cfg.get('post_training', {})
    probe_cfg = None
    if pt.get('run_probe', False):
        probe_cfg = OmegaConf.to_container(pt.probe, resolve=True)
        probe_cfg['img_size'] = cfg.img_size

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
            probe_cfg=probe_cfg,
            device=cfg.get('device', 'cpu'),
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

    _run_post_training(cfg, world_model.model, pl_logger)


def _run_post_training(cfg, model, pl_logger):
    """Run CEM planning eval after training and log to W&B."""
    pt = cfg.get('post_training', {})
    if not pt:
        return

    import wandb

    def _wandb_log(metrics: dict):
        if wandb.run is not None:
            wandb.log(metrics)

    # ── CEM planning evaluation ─────────────────────────────────────────────
    if pt.get('run_eval', False):
        logging.info('=== Post-training: CEM planning evaluation ===')
        from omegaconf import OmegaConf as OC
        from eval_wm import eval_model

        eval_cfg_raw = OmegaConf.to_container(pt.eval, resolve=True)
        eval_cfg = OC.create({
            'world': {
                'env_name':          eval_cfg_raw['env_name'],
                'num_envs':          eval_cfg_raw['num_envs'],
                'max_episode_steps': eval_cfg_raw['max_episode_steps'],
            },
            'seed':        eval_cfg_raw['seed'],
            'policy':      f"{cfg.output_model_name}/weights_epoch_{cfg.trainer.max_epochs:04d}.pt",
            'solver':      eval_cfg_raw['solver'],
            'plan_config': eval_cfg_raw['plan_config'],
            'dataset':     {'keys_to_cache': eval_cfg_raw['keys_to_cache']},
            'eval': {
                'num_eval':           eval_cfg_raw['num_eval'],
                'goal_offset_steps':  eval_cfg_raw['goal_offset_steps'],
                'eval_budget':        eval_cfg_raw['eval_budget'],
                'img_size':           cfg.img_size,
                'dataset_name':       eval_cfg_raw['dataset_name'],
                'callables':          eval_cfg_raw['callables'],
            },
            'device': cfg.device,
            'bf16':   False,
            'compile': False,
            'output': {'filename': 'eval_results.txt'},
        })

        video_dir = Path(swm.data.utils.get_cache_dir('checkpoints')) / cfg.output_model_name
        eval_metrics = eval_model(eval_cfg, model=model)

        scalar_metrics = {
            'eval/success_rate':    eval_metrics.get('success_rate', float('nan')),
            'eval/evaluation_time': eval_metrics.get('evaluation_time', float('nan')),
        }

        import wandb
        if wandb.run is not None:
            # Log scalars
            wandb.log(scalar_metrics)

            # Log rollout videos — cap at 10 to keep artifact size small
            videos = sorted(video_dir.glob('env_*.mp4'))
            successes = eval_metrics.get('episode_successes', [])
            video_log = {}
            for i, vid_path in enumerate(videos[:10]):
                success = bool(successes[i]) if i < len(successes) else None
                label = 'success' if success else 'fail' if success is not None else 'unknown'
                video_log[f'eval/rollout_{i:02d}_{label}'] = wandb.Video(
                    str(vid_path), fps=10, format='mp4'
                )
            if video_log:
                wandb.log(video_log)
                logging.info(f'Logged {len(video_log)} rollout videos to W&B')
        else:
            logging.info(scalar_metrics)

        logging.info(f"Eval success_rate: {eval_metrics.get('success_rate'):.1f}%")


if __name__ == '__main__':
    run()
