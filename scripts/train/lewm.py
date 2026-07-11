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
import torch.nn.functional as F
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
    lambd       = cfg.loss.sigreg.weight
    lambd_curv  = cfg.loss.get('curv',  {}).get('weight', 0.0)
    lambd_speed = cfg.loss.get('speed', {}).get('weight', 0.0)
    lambd_agir  = cfg.loss.get('agir',  {}).get('weight', 0.0)
    sigma_act   = cfg.loss.get('agir',  {}).get('sigma_act', 1.0)

    # LV-JEPA: β-KL annealing — linear warmup from beta_start to beta_end
    lv_cfg      = cfg.loss.get('lvjepa', {})
    beta_start  = lv_cfg.get('beta_start', 0.0)
    beta_end    = lv_cfg.get('beta_end',   0.0)
    anneal_epochs = max(1, lv_cfg.get('anneal_epochs', 1))
    frac        = min(1.0, getattr(self, 'current_epoch', 0) / anneal_epochs)
    beta        = beta_start + frac * (beta_end - beta_start)

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)
    emb     = output['emb']      # (B, T, D)
    act_emb = output['act_emb']  # (B, T, A)

    ctx_emb = emb[:, :ctx_len]       # (B, ctx, D)  — z_0 … z_{ctx-1}
    ctx_act = act_emb[:, :ctx_len]   # (B, ctx, A)
    tgt_emb = emb[:, n_preds:]       # (B, ctx, D)  — z_1 … z_{ctx}

    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))

    # Temporal straightening
    vel     = emb[:, 1:, :] - emb[:, :-1, :]                         # (B, T-1, D)
    cos_sim = F.cosine_similarity(vel[:, :-1, :], vel[:, 1:, :],
                                  dim=-1, eps=1e-6)                    # (B, T-2)
    output['curv_loss']  = (1.0 - cos_sim).mean()
    output['speed_loss'] = vel.norm(p=2, dim=-1).mean()

    # AGIR gate: opens speed penalty at physical contact events
    delta_act  = batch['action'][:, 1:] - batch['action'][:, :-1]    # (B, T-1, A)
    act_change = delta_act.norm(p=2, dim=-1).detach()                  # (B, T-1)
    gate       = torch.exp(-act_change / sigma_act)                    # (B, T-1) ∈ (0,1]
    lat_speed  = vel.norm(p=2, dim=-1)                                 # (B, T-1)
    output['agir_loss'] = (gate * lat_speed).mean()

    # LV-JEPA: posterior inference + reparameterisation trick.
    # w is computed whenever inference_net exists — cond_proj is sized for
    # concat(act_emb, w) and would crash if w=None but w_dim > 0.
    # The KL weight beta handles "on/off"; model structure does not.
    w = None
    output['kl_loss'] = torch.zeros(1, device=emb.device)
    if self.model.inference_net is not None:
        # q_φ(w_t | z_t, a_t, z_{t+1}) for each step in the context window
        mu_w, logvar_w = self.model.inference_net(ctx_emb, ctx_act, tgt_emb)
        eps = torch.randn_like(mu_w)
        w   = mu_w + (0.5 * logvar_w).exp() * eps     # (B, ctx, W)

        # Analytical KL(N(μ,σ²) || N(0,I)) = 0.5 * Σ(-1 - logvar + exp(logvar) + μ²)
        output['kl_loss'] = (
            0.5 * (-1.0 - logvar_w + logvar_w.exp() + mu_w.pow(2)).mean()
        )

        # Posterior health diagnostics — logged separately from losses
        with torch.no_grad():
            # KL per latent dimension: (W,) — reveals which dims carry information
            kl_per_dim = 0.5 * (
                -1.0 - logvar_w + logvar_w.exp() + mu_w.pow(2)
            ).mean(dim=(0, 1))                                    # (W,)
            sigma_w = (0.5 * logvar_w).exp()

            lv_diag = {
                # Number of dims with KL > 0.1 nats: tracks bottleneck utilisation
                f'{stage}/w_active_dims': (kl_per_dim > 0.1).sum().float(),
                # Mean posterior σ: near 1.0 = prior collapse, near 0 = overfit
                f'{stage}/w_sigma_mean': sigma_w.mean(),
                # Mean posterior ‖μ‖: should stay near 0 under KL pressure
                f'{stage}/w_mu_norm': mu_w.norm(p=2, dim=-1).mean(),
            }
            # Per-dimension KL: fingerprint of which dims are active
            for i, kl_d in enumerate(kl_per_dim):
                lv_diag[f'{stage}/w_kl_dim_{i}'] = kl_d

        self.log_dict(lv_diag, on_step=True, sync_dist=True)

    pred_emb = self.model.predict(ctx_emb, ctx_act, w)
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()

    output['loss'] = (
        output['pred_loss']
        + lambd       * output['sigreg_loss']
        + lambd_curv  * output['curv_loss']
        + lambd_speed * output['speed_loss']
        + lambd_agir  * output['agir_loss']
        + beta        * output['kl_loss']
    )

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True, sync_dist=True,
    )
    self.log(f'{stage}/kl_beta', beta, on_step=True, sync_dist=True)
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
    pl_logger = build_wandb_logger(cfg, run_dir)

    ckpt_dir = run_dir / 'lightning'
    last_ckpt = ckpt_dir / 'last.ckpt'

    pt = cfg.get('post_training', {})
    probe_cfg = None
    if pt.get('run_probe', False):
        probe_cfg = OmegaConf.to_container(pt.probe, resolve=True)
        probe_cfg['img_size'] = cfg.img_size

    callbacks = [
        # Quality checkpoints — ranked by val/loss, kept for model selection
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='epoch={epoch:04d}',
            monitor='validate/loss',
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=False,
            mode='min',
            verbose=True,
        ),
        # Resume checkpoint — overwrites last.ckpt every N steps so a
        # mid-epoch kill loses at most resume_every_n_steps of work
        ModelCheckpoint(
            dirpath=ckpt_dir,
            every_n_train_steps=cfg.checkpointing.resume_every_n_steps,
            save_top_k=0,
            save_last=True,
            enable_version_counter=False,
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
