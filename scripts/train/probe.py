"""Offline linear probing of world model encoder representations.

For each representation level (encoder CLS, mean patches, projector output)
fit a Ridge regressor on frozen embeddings to predict each state variable,
and report R² on the held-out validation split.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utils import get_img_preprocessor, setup_run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_batch(model, pixels: torch.Tensor, device: str) -> dict[str, np.ndarray]:
    """Return embeddings at every probe level for one batch of pixels.

    pixels: (B, 1, C, H, W) — single-frame windows from the dataset.
    Returns dict {level_name: (B, D) float32 ndarray}.
    """
    x = pixels.squeeze(1).to(device)  # (B, C, H, W)
    with torch.no_grad():
        out = model.encoder(x, interpolate_pos_encoding=True)
        hidden = out.last_hidden_state  # (B, L+1, D)
        cls = hidden[:, 0]             # CLS token
        patches_mean = hidden[:, 1:].mean(dim=1)
        projected = model.projector(cls)

    def to_np(t):
        return t.float().cpu().numpy()

    return {
        'encoder_cls': to_np(cls),
        'encoder_patches': to_np(patches_mean),
        'projector': to_np(projected),
    }


def encode_dataset(
    model, loader: DataLoader, probe_targets: list[str], device: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode the full dataloader.

    Returns:
        embeddings: {level_name: (N, D)}
        targets:    {variable_name: (N, *)}
    """
    level_buffers: dict[str, list] = {}
    target_buffers: dict[str, list] = {t: [] for t in probe_targets}

    for batch in loader:
        embs = _encode_batch(model, batch['pixels'], device)
        for level, arr in embs.items():
            level_buffers.setdefault(level, []).append(arr)
        for t in probe_targets:
            if t in batch:
                v = batch[t].squeeze(1)   # (B, 1, *) → (B, *)
                if v.dim() > 2:
                    v = v.squeeze(-1)     # (B, 1) → (B,)
                target_buffers[t].append(v.float().cpu().numpy())

    embeddings = {k: np.concatenate(v, axis=0) for k, v in level_buffers.items()}
    targets = {k: np.concatenate(v, axis=0) for k, v in target_buffers.items() if v}
    return embeddings, targets


# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────

def _valid_mask(emb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Boolean mask of rows with no NaN/inf in either emb or target."""
    emb_ok = np.isfinite(emb).all(axis=1)
    tgt = target if target.ndim > 1 else target[:, None]
    tgt_ok = np.isfinite(tgt).all(axis=1)
    return emb_ok & tgt_ok


def probe_level(
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, dict]:
    """Fit a Ridge probe per target variable and return R² scores."""
    # float64 avoids overflow in sklearn's SVD solver on large matrices
    train_emb = train_emb.astype(np.float64)
    val_emb = val_emb.astype(np.float64)

    # fit scaler on finite train embeddings only
    emb_finite = np.isfinite(train_emb).all(axis=1)
    scaler = StandardScaler()
    scaler.fit(train_emb[emb_finite])
    X_train = scaler.transform(train_emb)
    X_val = scaler.transform(val_emb)

    results = {}
    for name, y_train in train_targets.items():
        y_val = val_targets[name]
        y_train = y_train.astype(np.float64)
        y_val = y_val.astype(np.float64)

        # drop NaN/inf rows independently for train and val
        tr_mask = _valid_mask(X_train, y_train)
        va_mask = _valid_mask(X_val, y_val)
        n_drop_tr = (~tr_mask).sum()
        n_drop_va = (~va_mask).sum()
        if n_drop_tr:
            logging.debug(f'  {name}: dropped {n_drop_tr} NaN train rows')
        if n_drop_va:
            logging.debug(f'  {name}: dropped {n_drop_va} NaN val rows')

        probe = Ridge(alpha=alpha)
        probe.fit(X_train[tr_mask], y_train[tr_mask])
        y_pred = probe.predict(X_val[va_mask])

        # per-dim and mean R²
        y_v = y_val[va_mask]
        if y_v.ndim == 1 or y_v.shape[1] == 1:
            r2 = float(r2_score(y_v, y_pred))
            results[name] = {'mean': r2}
        else:
            per_dim = [
                float(r2_score(y_v[:, i], y_pred[:, i]))
                for i in range(y_v.shape[1])
            ]
            results[name] = {
                'mean': float(np.mean(per_dim)),
                'per_dim': per_dim,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API (callable without Hydra)
# ─────────────────────────────────────────────────────────────────────────────

def probe_model(model, probe_cfg, device: str) -> dict[str, dict]:
    """Run linear probing on a frozen model and return results dict.

    probe_cfg must have:
        dataset_name, keys_to_load, num_steps, frameskip,
        img_size, train_split, seed, ridge_alpha,
        probe_levels, probe_targets
    Accepts both OmegaConf nodes and plain dicts.
    """
    from omegaconf import OmegaConf as _OC
    if hasattr(probe_cfg, '_metadata'):  # OmegaConf node
        cfg_d = _OC.to_container(probe_cfg, resolve=True)
    else:
        cfg_d = dict(probe_cfg)

    probe_targets = list(cfg_d['probe_targets'])
    probe_levels  = list(cfg_d['probe_levels'])

    was_training = model.training
    saved_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    model.eval().requires_grad_(False).to(device)

    dataset = swm.data.load_dataset(
        cfg_d['dataset_name'],
        keys_to_load=cfg_d['keys_to_load'],
        num_steps=cfg_d.get('num_steps', 1),
        frameskip=cfg_d.get('frameskip', 1),
    )
    dataset.transform = get_img_preprocessor('pixels', 'pixels', cfg_d['img_size'])

    rng = torch.Generator().manual_seed(cfg_d['seed'])
    train_set, val_set = spt.data.random_split(
        dataset, [cfg_d['train_split'], 1 - cfg_d['train_split']], generator=rng
    )
    train_loader = DataLoader(train_set, batch_size=cfg_d.get('batch_size', 512),
                              shuffle=False, num_workers=2, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=cfg_d.get('batch_size', 512),
                              shuffle=False, num_workers=2, drop_last=False)

    try:
        logging.info(f'[probe] encoding {len(train_set):,} train / {len(val_set):,} val frames...')
        train_embs, train_targets = encode_dataset(model, train_loader, probe_targets, device)
        val_embs,   val_targets   = encode_dataset(model, val_loader,   probe_targets, device)

        results: dict[str, dict] = {}
        for level in probe_levels:
            if level not in train_embs:
                logging.warning(f'[probe] level {level!r} not found, skipping.')
                continue
            logging.info(f'[probe] probing level: {level}')
            results[level] = probe_level(
                train_embs[level], val_embs[level],
                train_targets, val_targets,
                alpha=cfg_d['ridge_alpha'],
            )

        # print table
        col_w = max(len(t) for t in probe_targets) + 2
        header = f"{'variable':<{col_w}}" + ''.join(f'{l:>18}' for l in probe_levels)
        sep = '─' * len(header)
        print(f'\n{sep}\nLinear Probing Results  (R²)\n{sep}')
        print(header)
        print(sep)
        for target in probe_targets:
            row = f'{target:<{col_w}}'
            for level in probe_levels:
                r2 = results.get(level, {}).get(target, {}).get('mean', float('nan'))
                row += f'{r2:>18.4f}'
            print(row)
        print(sep)

        return results
    finally:
        # Restore model training state so probing mid-training doesn't break backprop
        for n, p in model.named_parameters():
            p.requires_grad_(saved_grad.get(n, True))
        if was_training:
            model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point (standalone usage)
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path='./config', config_name='probe')
def run(cfg):
    device = cfg.get('device', 'cpu')

    logging.info(f'Loading model: {cfg.model_name}')
    model = swm.wm.utils.load_pretrained(cfg.model_name)
    model.to(device)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    probe_cfg = {
        'dataset_name':  dataset_cfg.pop('name'),
        'keys_to_load':  dataset_cfg.pop('keys_to_load'),
        **dataset_cfg,
        'img_size':      cfg.img_size,
        'train_split':   cfg.train_split,
        'seed':          cfg.seed,
        'ridge_alpha':   cfg.ridge_alpha,
        'batch_size':    cfg.batch_size,
        'probe_levels':  list(cfg.probe_levels),
        'probe_targets': list(cfg.probe_targets),
    }

    results = probe_model(model, probe_cfg, device)

    run_dir = setup_run_dir(cfg)
    out_path = run_dir / 'probe_results.json'
    with open(out_path, 'w') as f:
        json.dump(
            {
                'model':      cfg.model_name,
                'dataset':    probe_cfg['dataset_name'],
                'n_train':    int(0.9 * len(swm.data.load_dataset(probe_cfg['dataset_name'],
                                  keys_to_load=['pixels'], num_steps=1, frameskip=1))),
                'ridge_alpha': cfg.ridge_alpha,
                'results':    results,
            },
            f,
            indent=2,
        )
    logging.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    run()
