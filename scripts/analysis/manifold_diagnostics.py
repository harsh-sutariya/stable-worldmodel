#!/usr/bin/env python3
"""
Three-experiment latent manifold failure analysis.

Exp 1 — Intra/Inter trajectory variance decomposition
Exp 2 — CEM score variance (planning SNR test)
Exp 3 — Kinematic sensitivity (physical vs latent velocity Pearson r)

Usage:
    python diagnostics.py --ckpt <name> [--tag <label>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

ROOT = Path('/Users/harsh/projects/personal/swm/stable-worldmodel')
sys.path.insert(0, str(ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET    = 'galilai-group/lewm-pusht'
IMG_SIZE   = 56
TRAJ_LEN   = 35
FRAMESKIP  = 5
NUM_TRAJ   = 300     # for exp 1 and 3
BATCH_SIZE = 32
DEVICE     = 'mps'
K_SAMPLES  = 300     # CEM candidates for exp 2
N_PLANNING = 50      # number of starting states for exp 2
H_PLAN     = 5       # CEM horizon (receding)
CTX_LEN    = 3       # predictor context length
ACTION_DIM = 10      # frameskip(5) × raw_action_dim(2)


def pixel_transform(img_size):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source='pixels', target='pixels'),
        spt.data.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


@torch.no_grad()
def load_trajectories(model, num_traj, device):
    """Return Z (N,T,D) projector embeddings and S (N,T,7) physical states."""
    transform = pixel_transform(IMG_SIZE)
    dataset = swm.data.load_dataset(
        DATASET, num_steps=TRAJ_LEN, frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'state'],
    )
    dataset.transform = transform
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
    )

    Z_list, S_list = [], []
    collected = 0
    model.to(device).eval()
    for batch in loader:
        pixels = batch['pixels'].to(device)  # (B, T, C, H, W)
        state  = batch['state'].float()
        B, T, C, H, W = pixels.shape
        flat   = pixels.view(B * T, C, H, W)
        enc    = model.encoder(flat)
        cls    = enc.last_hidden_state[:, 0]
        emb    = model.projector(cls).view(B, T, -1).cpu().float()
        Z_list.append(emb.numpy())
        S_list.append(state.numpy())
        collected += B
        if collected >= num_traj:
            break

    Z = np.concatenate(Z_list)[:num_traj]
    S = np.concatenate(S_list)[:num_traj]
    return Z, S


# ── Experiment 1 ──────────────────────────────────────────────────────────────

def exp1_variance_decomposition(Z):
    """
    Z: (N, T, D)
    Returns intra_var, inter_var, ratio.
    """
    N, T, D = Z.shape
    centroids = Z.mean(axis=1)                        # (N, D) — episode mean
    # intra: mean squared deviation of frames from their episode centroid
    intra_per_ep = np.mean(
        np.sum((Z - centroids[:, None, :]) ** 2, axis=-1), axis=1
    )                                                 # (N,)
    intra_var = float(intra_per_ep.mean())

    # inter: mean squared deviation of centroids from the global mean
    global_centroid = centroids.mean(axis=0)          # (D,)
    inter_per_ep = np.sum(
        (centroids - global_centroid[None, :]) ** 2, axis=-1
    )                                                 # (N,)
    inter_var = float(inter_per_ep.mean())

    ratio = inter_var / (intra_var + 1e-12)
    return intra_var, inter_var, ratio


# ── Experiment 2 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def exp2_cem_snr(model, device):
    """
    Simulate CEM first planning step: sample K random action sequences, roll
    out H_PLAN steps from a real context, score against a real goal embedding.

    Sliding-window autoregressive rollout:
      At step t, emb_window holds the last CTX_LEN predicted frames.
      We predict one step ahead using emb_window + act_window[t:t+CTX_LEN].

    Returns score_std, score_range, score_mean, snr.
    """
    transform = pixel_transform(IMG_SIZE)
    total_steps = CTX_LEN + H_PLAN + 25   # context + rollout + goal offset
    dataset = swm.data.load_dataset(
        DATASET, num_steps=total_steps, frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'action'],
    )
    dataset.transform = transform
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=N_PLANNING, shuffle=True, num_workers=0, drop_last=True
    )
    batch = next(iter(loader))
    pixels  = batch['pixels'].to(device)   # (B, total_steps, C, H, W)
    actions = batch['action'].to(device)   # (B, total_steps, ...)

    B      = pixels.size(0)
    T_full = pixels.size(1)

    # ── encode all frames ────────────────────────────────────────────────────
    flat = pixels.view(B * T_full, *pixels.shape[2:])
    cls  = model.encoder(flat).last_hidden_state[:, 0]
    emb  = model.projector(cls).view(B, T_full, -1)   # (B, T_full, D)

    ctx_emb  = emb[:, :CTX_LEN]             # (B, CTX, D)
    goal_emb = emb[:, CTX_LEN + H_PLAN + 24]  # (B, D) — goal at +25 steps

    # ── action stats from real data ──────────────────────────────────────────
    act_raw = actions.view(B, T_full, -1).float()
    # normalise to ACTION_DIM
    d = act_raw.shape[-1]
    if d < ACTION_DIM:
        pad = torch.zeros(B, T_full, ACTION_DIM - d, device=device)
        act_raw = torch.cat([act_raw, pad], dim=-1)
    else:
        act_raw = act_raw[..., :ACTION_DIM]
    act_mean = act_raw.mean().item()
    act_std  = max(act_raw.std().item(), 1e-3)

    # ── sample K random action sequences of length CTX_LEN + H_PLAN ─────────
    # At rollout step t we use actions[t : t+CTX_LEN], so we need
    # indices 0 … CTX_LEN + H_PLAN - 1 total.
    T_plan = CTX_LEN + H_PLAN                         # 8
    rand_act = (torch.randn(B * K_SAMPLES, T_plan, ACTION_DIM, device=device)
                * act_std + act_mean)
    rand_act_emb = model.action_encoder(rand_act)      # (B*K, T_plan, A_emb)

    # ── expand context to B*K ────────────────────────────────────────────────
    # repeat_interleave keeps sample groups together: [ep0_s0, ep0_s1, ..., ep1_s0, ...]
    emb_win = ctx_emb.repeat_interleave(K_SAMPLES, dim=0).clone()  # (B*K, CTX, D)

    # ── sliding-window rollout ────────────────────────────────────────────────
    for t in range(H_PLAN):
        act_win = rand_act_emb[:, t:t + CTX_LEN]      # (B*K, CTX, A_emb)
        next_e  = model.predict(emb_win, act_win)[:, -1:]  # (B*K, 1, D)
        emb_win = torch.cat([emb_win[:, 1:], next_e], dim=1)

    final_emb = emb_win[:, -1].view(B, K_SAMPLES, -1)  # (B, K, D)

    # ── score and statistics ─────────────────────────────────────────────────
    goal_exp = goal_emb.unsqueeze(1).expand_as(final_emb)
    scores   = F.mse_loss(final_emb, goal_exp, reduction='none').sum(dim=-1)  # (B, K)

    score_std   = scores.std(dim=1).mean().item()
    score_range = (scores.max(dim=1).values - scores.min(dim=1).values).mean().item()
    score_mean  = scores.mean().item()
    snr         = score_mean / (score_std + 1e-12)

    return score_std, score_range, score_mean, snr


# ── Experiment 3 ──────────────────────────────────────────────────────────────

def exp3_kinematic_sensitivity(Z, S):
    """
    Z: (N, T, D)  projector embeddings
    S: (N, T, 7)  physical state [block_x, block_y, goal_x, goal_y, block_angle, agent_x, agent_y]

    Compute step-wise:
      physical_vel = ||s_{t+1} - s_t||_2   (using block_x, block_y, agent_x, agent_y)
      latent_vel   = ||z_{t+1} - z_t||_2

    Return Pearson r and its p-value.
    """
    # Physical: use block + agent position (4 dims most dynamically informative)
    pos_dims = [0, 1, 5, 6]    # block_x, block_y, agent_x, agent_y
    S_pos = S[:, :, pos_dims]  # (N, T, 4)

    phys_vel = np.linalg.norm(S_pos[:, 1:, :] - S_pos[:, :-1, :], axis=-1).ravel()   # (N*(T-1),)
    lat_vel  = np.linalg.norm(Z[:, 1:, :] - Z[:, :-1, :], axis=-1).ravel()           # (N*(T-1),)

    r, p = pearsonr(phys_vel, lat_vel)

    # also compute block-only velocity for focus on task-relevant dynamics
    block_vel = np.linalg.norm(S[:, 1:, :2] - S[:, :-1, :2], axis=-1).ravel()
    r_block, p_block = pearsonr(block_vel, lat_vel)

    return float(r), float(p), float(r_block), float(p_block)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag',  default=None)
    p.add_argument('--device', default=DEVICE)
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()

    tag = args.tag or args.ckpt
    print(f'\n{"="*60}')
    print(f'Model: {tag}')
    print(f'{"="*60}')

    print('Loading model...')
    model = load_pretrained(args.ckpt)
    model.to(args.device).eval()

    # ── Exp 1 ────────────────────────────────────────────────────────────────
    print('\n[Exp 1] Loading trajectories for variance decomposition...')
    Z, S = load_trajectories(model, NUM_TRAJ, args.device)
    intra, inter, ratio = exp1_variance_decomposition(Z)
    print(f'  intra_var = {intra:.4f}')
    print(f'  inter_var = {inter:.4f}')
    print(f'  ratio (inter/intra) = {ratio:.4f}')

    # ── Exp 2 ────────────────────────────────────────────────────────────────
    print('\n[Exp 2] CEM score variance (SNR test)...')
    score_std, score_range, score_mean, snr = exp2_cem_snr(model, args.device)
    print(f'  score_mean  = {score_mean:.6f}')
    print(f'  score_std   = {score_std:.6f}')
    print(f'  score_range = {score_range:.6f}')
    print(f'  SNR (mean/std) = {snr:.4f}')

    # ── Exp 3 ────────────────────────────────────────────────────────────────
    print('\n[Exp 3] Kinematic sensitivity (physical vs latent velocity)...')
    r, p_val, r_block, p_block = exp3_kinematic_sensitivity(Z, S)
    print(f'  Pearson r (all-pos):   {r:.4f}  (p={p_val:.2e})')
    print(f'  Pearson r (block-pos): {r_block:.4f}  (p={p_block:.2e})')

    result = {
        'tag': tag,
        'ckpt': args.ckpt,
        'exp1': {'intra_var': intra, 'inter_var': inter, 'ratio': ratio},
        'exp2': {'score_mean': score_mean, 'score_std': score_std,
                 'score_range': score_range, 'snr': snr},
        'exp3': {'pearson_r_all': r, 'p_all': p_val,
                 'pearson_r_block': r_block, 'p_block': p_block},
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nSaved to {args.out}')

    return result


if __name__ == '__main__':
    main()
