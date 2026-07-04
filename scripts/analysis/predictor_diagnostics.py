#!/usr/bin/env python3
"""
Predictor and information collapse diagnostics.

Exp 4 — Action sensitivity: does the predictor's output change when actions change?
Exp 5 — Residual magnitude ratio: predicted step vs true step size.
Exp 6 — Trajectory-to-global volume ratio: local vs global covariance trace.

Usage:
    python diagnostics2.py --ckpt <name> [--tag <label>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/Users/harsh/projects/personal/swm/stable-worldmodel')
sys.path.insert(0, str(ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained

DATASET    = 'galilai-group/lewm-pusht'
IMG_SIZE   = 56
FRAMESKIP  = 5
CTX_LEN    = 3
ACTION_DIM = 10      # frameskip(5) × raw_action_dim(2)
BATCH_SIZE = 256     # large batch for stable statistics
NUM_TRAJ   = 300     # for exp6
TRAJ_LEN   = 35
DEVICE     = 'mps'


def pixel_transform():
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source='pixels', target='pixels'),
        spt.data.transforms.Resize(IMG_SIZE, source='pixels', target='pixels'),
    )


def load_context_batch(device, num_steps=CTX_LEN + 1, batch_size=BATCH_SIZE):
    """Load one large batch of CTX_LEN+1 frame windows with actions."""
    dataset = swm.data.load_dataset(
        DATASET, num_steps=num_steps, frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'action'],
    )
    dataset.transform = pixel_transform()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    batch = next(iter(loader))
    pixels  = batch['pixels'].to(device)   # (B, num_steps, C, H, W)
    actions = batch['action'].to(device)   # (B, num_steps, ...)
    return pixels, actions


@torch.no_grad()
def encode_ctx(model, pixels, device):
    """Encode CTX_LEN frames → projector embeddings (B, CTX, D)."""
    B, T = pixels.shape[:2]
    flat = pixels.view(B * T, *pixels.shape[2:])
    cls  = model.encoder(flat).last_hidden_state[:, 0]
    emb  = model.projector(cls).view(B, T, -1)
    return emb


def prep_actions(actions, B, T, device):
    """Normalise action tensor to (B, T, ACTION_DIM)."""
    act = actions.view(B, T, -1).float()
    d = act.shape[-1]
    if d < ACTION_DIM:
        act = torch.cat([act, torch.zeros(B, T, ACTION_DIM - d, device=device)], dim=-1)
    else:
        act = act[..., :ACTION_DIM]
    return act


# ── Experiment 4: Action Sensitivity ─────────────────────────────────────────

@torch.no_grad()
def exp4_action_sensitivity(model, device):
    """
    Measure how much the predictor's output changes when actions are replaced
    with out-of-distribution noise.  Three conditions:
      real    — actual dataset actions
      shuffle — batch-shuffled real actions (same distribution, broken causal link)
      ood     — Gaussian noise 5× the action std (strongly OOD)

    Divergence = ||pred_real - pred_perturbed||_2  (mean over batch)
    Also reports identity residual = ||pred - ctx[:, -1]||_2 to detect identity collapse.
    """
    pixels, actions = load_context_batch(device)
    B = pixels.size(0)

    # encode context (CTX_LEN frames) and ground truth next frame
    emb_full = encode_ctx(model, pixels, device)        # (B, CTX+1, D)
    ctx_emb  = emb_full[:, :CTX_LEN]                   # (B, CTX, D)
    last_ctx = ctx_emb[:, -1]                           # (B, D)  — z_t

    act_raw = prep_actions(actions, B, CTX_LEN + 1, device)[:, :CTX_LEN]  # (B, CTX, 10)
    act_std = max(act_raw.std().item(), 1e-3)

    # real action embedding
    act_emb_real    = model.action_encoder(act_raw)                          # (B, CTX, A)

    # shuffled: permute batch dimension → same marginal, broken conditional
    perm = torch.randperm(B, device=device)
    act_emb_shuffle = model.action_encoder(act_raw[perm])                    # (B, CTX, A)

    # OOD: Gaussian with 5× the real action std
    act_ood = torch.randn_like(act_raw) * act_std * 5
    act_emb_ood = model.action_encoder(act_ood)                              # (B, CTX, A)

    # predictions
    pred_real    = model.predict(ctx_emb, act_emb_real)[:, -1]    # (B, D)
    pred_shuffle = model.predict(ctx_emb, act_emb_shuffle)[:, -1]
    pred_ood     = model.predict(ctx_emb, act_emb_ood)[:, -1]

    div_shuffle = (pred_real - pred_shuffle).norm(p=2, dim=-1).mean().item()
    div_ood     = (pred_real - pred_ood).norm(p=2, dim=-1).mean().item()

    # identity residual: how far pred moved from z_t
    id_res_real    = (pred_real    - last_ctx).norm(p=2, dim=-1).mean().item()
    id_res_shuffle = (pred_shuffle - last_ctx).norm(p=2, dim=-1).mean().item()
    id_res_ood     = (pred_ood     - last_ctx).norm(p=2, dim=-1).mean().item()

    return {
        'div_shuffle':     div_shuffle,
        'div_ood':         div_ood,
        'id_res_real':     id_res_real,
        'id_res_shuffle':  id_res_shuffle,
        'id_res_ood':      id_res_ood,
    }


# ── Experiment 5: Predictor Identity Metric ──────────────────────────────────

@torch.no_grad()
def exp5_residual_ratio(model, device):
    """
    Compare the magnitude of the predictor's output residual against the true
    latent step size.

      true_step  = ||z_{t+1} - z_t||_2    (actual last step in the window)
      pred_step  = ||pred_{t+1} - z_t||_2  (predictor's estimated step)
      ratio      = mean(pred_step) / mean(true_step)

    ratio ≈ 1 → dynamics correctly modelled.
    ratio → 0 → predictor outputs near z_t (identity collapse).

    Also reports the Pearson r between pred_step and true_step per sample
    to check whether the predictor at least ranks step magnitudes correctly.
    """
    from scipy.stats import pearsonr

    pixels, actions = load_context_batch(device)
    B = pixels.size(0)

    emb_full = encode_ctx(model, pixels, device)        # (B, CTX+1, D)
    ctx_emb  = emb_full[:, :CTX_LEN]                   # (B, CTX, D)
    z_next   = emb_full[:, CTX_LEN]                    # (B, D)  — true z_{t+1}
    z_t      = ctx_emb[:, -1]                          # (B, D)  — z_t

    act_raw  = prep_actions(actions, B, CTX_LEN + 1, device)[:, :CTX_LEN]
    act_emb  = model.action_encoder(act_raw)

    pred_next = model.predict(ctx_emb, act_emb)[:, -1]  # (B, D)

    true_step = (z_next  - z_t).norm(p=2, dim=-1).cpu().numpy()  # (B,)
    pred_step = (pred_next - z_t).norm(p=2, dim=-1).cpu().numpy() # (B,)

    ratio = float(pred_step.mean() / (true_step.mean() + 1e-12))
    r_step, p_step = pearsonr(true_step, pred_step)

    # MSE between true and predicted next embedding (prediction error)
    mse = F.mse_loss(pred_next, z_next).item()

    return {
        'true_step_mean': float(true_step.mean()),
        'pred_step_mean': float(pred_step.mean()),
        'ratio':          ratio,
        'pearson_r_step': float(r_step),
        'p_step':         float(p_step),
        'pred_mse':       mse,
    }


# ── Experiment 6: Trajectory-to-Global Volume Ratio ──────────────────────────

@torch.no_grad()
def load_trajectories(model, device):
    """Return Z (N, T, D) projector embeddings."""
    transform = pixel_transform()
    dataset = swm.data.load_dataset(
        DATASET, num_steps=TRAJ_LEN, frameskip=FRAMESKIP,
        keys_to_load=['pixels'],
    )
    dataset.transform = transform
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False
    )
    Z_list = []
    collected = 0
    for batch in loader:
        pixels = batch['pixels'].to(device)
        B, T = pixels.shape[:2]
        flat = pixels.view(B * T, *pixels.shape[2:])
        cls  = model.encoder(flat).last_hidden_state[:, 0]
        emb  = model.projector(cls).view(B, T, -1).cpu().float()
        Z_list.append(emb.numpy())
        collected += B
        if collected >= NUM_TRAJ:
            break
    return np.concatenate(Z_list)[:NUM_TRAJ]


def exp6_volume_ratio(Z):
    """
    Z: (N, T, D)

    global_trace = Tr(Cov(Z_flat))  = sum of per-dim variances across all N×T frames
    local_trace  = mean over N of Tr(Cov(Z[n]))  = per-trajectory variance sum
    ratio        = local_trace / global_trace

    ratio ≈ 1  → trajectories span the full global space (ideal)
    ratio → 0  → trajectories are degenerate micro-clusters (archipelago)

    Also reports the effective intrinsic dimensionality of trajectories
    (participation ratio of local covariance eigenspectrum).
    """
    N, T, D = Z.shape
    Z_flat = Z.reshape(-1, D)

    # global
    global_trace = float(np.var(Z_flat, axis=0).sum())

    # local per trajectory
    local_traces = np.array([np.var(Z[n], axis=0).sum() for n in range(N)])
    local_trace_mean = float(local_traces.mean())
    local_trace_std  = float(local_traces.std())

    ratio = local_trace_mean / (global_trace + 1e-12)

    # participation ratio of local covariance (mean over trajectories)
    # PR = (sum eigenvalues)^2 / sum(eigenvalues^2) — effective dimension
    pr_list = []
    for n in range(min(N, 100)):   # cap at 100 for speed
        cov = np.cov(Z[n].T)       # (D, D)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        s = eigvals.sum()
        if s > 1e-10:
            pr_list.append((s ** 2) / (np.sum(eigvals ** 2) + 1e-20))
    local_pr_mean = float(np.mean(pr_list)) if pr_list else float('nan')

    # global PR for reference
    cov_global = np.cov(Z_flat.T)
    ev_global  = np.linalg.eigvalsh(cov_global)
    ev_global  = np.maximum(ev_global, 0)
    s = ev_global.sum()
    global_pr  = float((s ** 2) / (np.sum(ev_global ** 2) + 1e-20)) if s > 1e-10 else float('nan')

    return {
        'global_trace':      global_trace,
        'local_trace_mean':  local_trace_mean,
        'local_trace_std':   local_trace_std,
        'ratio':             ratio,
        'local_pr_mean':     local_pr_mean,
        'global_pr':         global_pr,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',   required=True)
    p.add_argument('--tag',    default=None)
    p.add_argument('--device', default=DEVICE)
    p.add_argument('--out',    type=Path, default=None)
    args = p.parse_args()

    tag = args.tag or args.ckpt
    print(f'\n{"="*60}')
    print(f'Model: {tag}')
    print(f'{"="*60}')

    model = load_pretrained(args.ckpt)
    model.to(args.device).eval()

    # ── Exp 4 ────────────────────────────────────────────────────────────────
    print('\n[Exp 4] Action sensitivity (predictor nullspace test)...')
    r4 = exp4_action_sensitivity(model, args.device)
    print(f'  div (real vs shuffle):      {r4["div_shuffle"]:.6f}')
    print(f'  div (real vs OOD ×5σ):      {r4["div_ood"]:.6f}')
    print(f'  identity residual (real):   {r4["id_res_real"]:.6f}')
    print(f'  identity residual (shuffle):{r4["id_res_shuffle"]:.6f}')
    print(f'  identity residual (OOD):    {r4["id_res_ood"]:.6f}')

    # ── Exp 5 ────────────────────────────────────────────────────────────────
    print('\n[Exp 5] Predictor residual magnitude ratio...')
    r5 = exp5_residual_ratio(model, args.device)
    print(f'  true_step_mean:  {r5["true_step_mean"]:.6f}')
    print(f'  pred_step_mean:  {r5["pred_step_mean"]:.6f}')
    print(f'  ratio:           {r5["ratio"]:.6f}')
    print(f'  Pearson r (step magnitudes): {r5["pearson_r_step"]:.4f}  (p={r5["p_step"]:.2e})')
    print(f'  pred MSE:        {r5["pred_mse"]:.6f}')

    # ── Exp 6 ────────────────────────────────────────────────────────────────
    print('\n[Exp 6] Trajectory-to-global volume ratio...')
    Z = load_trajectories(model, args.device)
    r6 = exp6_volume_ratio(Z)
    print(f'  global_trace:       {r6["global_trace"]:.4f}')
    print(f'  local_trace_mean:   {r6["local_trace_mean"]:.4f}  (±{r6["local_trace_std"]:.4f})')
    print(f'  ratio:              {r6["ratio"]:.6f}')
    print(f'  local PR (eff dim): {r6["local_pr_mean"]:.2f}')
    print(f'  global PR (eff dim):{r6["global_pr"]:.2f}')

    result = {'tag': tag, 'ckpt': args.ckpt, 'exp4': r4, 'exp5': r5, 'exp6': r6}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved to {args.out}')

    return result


if __name__ == '__main__':
    main()
