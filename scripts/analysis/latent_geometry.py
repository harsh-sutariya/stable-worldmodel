#!/usr/bin/env python3
"""
Latent Geometry Profiling — LeWM PushT Vanilla Baseline

Quantitatively characterises the Riemannian structure of the learned latent
manifold to assess whether Euclidean distance is a reliable proxy for
temporal/physical distance.

Metrics
-------
1. Curvature distribution   — cos(v_t, v_{t+1}), v_t = z_{t+1} − z_t
2. Latent speed             — ‖v_t‖₂ uniformity (CV = σ/μ)
3. Participation ratio      — effective dimensionality of the embedding cloud
4. MSD scaling law          — ‖z_{t+Δt} − z_t‖₂² vs Δt, power-law fit
5. Geodesic efficiency      — chord / path-length ratio per trajectory
6. Contact correlation      — curvature at effector–block contact events

State layout (7-dim, pixel coords on 512×512 canvas):
  [block_x, block_y, goal_x*, goal_y*, block_angle, agent_x, agent_y]
  * goal_x, goal_y are episode-fixed target positions
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained

# ── defaults ──────────────────────────────────────────────────────────────────
CKPT        = 'lewm_pusht/weights_epoch_0010.pt'
DATASET     = 'galilai-group/lewm-pusht'
IMG_SIZE    = 56
TRAJ_LEN    = 35      # steps per window (× frameskip=5 → 175 raw frames; episodes are 40 steps)
FRAMESKIP   = 5
NUM_TRAJ    = 400
BATCH_SIZE  = 32
DEVICE      = 'mps'
CONTACT_THRESH = 40.0   # px/step block velocity; fast push (top ~half) vs slow/approach


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',          default=CKPT)
    p.add_argument('--num-traj',      type=int,   default=NUM_TRAJ)
    p.add_argument('--traj-len',      type=int,   default=TRAJ_LEN)
    p.add_argument('--batch-size',    type=int,   default=BATCH_SIZE)
    p.add_argument('--device',        default=DEVICE)
    p.add_argument('--contact-thresh', type=float, default=CONTACT_THRESH)
    p.add_argument('--out-dir',       type=Path,
                   default=Path(ROOT) / 'outputs' / 'latent_geometry')
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def pixel_transform(img_size):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source='pixels', target='pixels'),
        spt.data.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


@torch.no_grad()
def extract_embeddings(model, args):
    """
    Load trajectory windows and encode every frame with the frozen encoder.

    Returns
    -------
    Z      : (N, T, D)   — CLS-token embeddings
    states : (N, T, 7)   — raw physical states (pixel coords)
    """
    transform = pixel_transform(args.img_size if hasattr(args, 'img_size') else IMG_SIZE)

    dataset = swm.data.load_dataset(
        DATASET,
        num_steps=args.traj_len,
        frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'state'],
    )
    dataset.transform = transform

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    Z_list, S_list = [], []
    collected = 0

    device = args.device
    model.to(device)

    for batch in loader:
        pixels = batch['pixels'].to(device)   # (B, T, C, H, W)
        state  = batch['state'].float()       # (B, T, 7)

        B, T, C, H, W = pixels.shape
        flat = pixels.view(B * T, C, H, W)

        enc_out = model.encoder(flat)          # HF BaseModelOutputWithPooling
        hs      = enc_out.last_hidden_state    # (B*T, N_patches+1, D)
        cls     = hs[:, 0, :]                 # CLS token  (B*T, D)
        cls     = cls.view(B, T, -1).cpu().float()

        Z_list.append(cls.numpy())
        S_list.append(state.numpy())

        collected += B
        print(f'\r  encoded {min(collected, args.num_traj)}/{args.num_traj}', end='', flush=True)
        if collected >= args.num_traj:
            break

    print()
    Z = np.concatenate(Z_list, axis=0)[:args.num_traj]   # (N, T, D)
    S = np.concatenate(S_list, axis=0)[:args.num_traj]   # (N, T, 7)
    return Z, S


# ── metrics ───────────────────────────────────────────────────────────────────

def curvature_and_speed(Z):
    """
    v_t = z_{t+1} − z_t                     velocity in latent space
    κ_t = cos(v_{t-1}, v_t) ∈ [−1, 1]      turning angle (curvature proxy)
    s_t = ‖v_t‖₂                             latent speed

    Returns
    -------
    vel        : (N, T-1, D)
    curvature  : (N, T-2)   — cosine similarity of consecutive velocities
    speed      : (N, T-1)
    """
    vel   = np.diff(Z, axis=1)                      # (N, T-1, D)
    speed = np.linalg.norm(vel, axis=-1)            # (N, T-1)

    v0   = vel[:, :-1, :]                           # (N, T-2, D)
    v1   = vel[:, 1:,  :]
    dot  = (v0 * v1).sum(-1)
    n0   = np.linalg.norm(v0, axis=-1)
    n1   = np.linalg.norm(v1, axis=-1)
    denom = n0 * n1
    kappa = np.where(denom > 1e-9, dot / denom, 0.0)   # (N, T-2)

    return vel, kappa, speed


def participation_ratio(Z):
    """
    PR = (Σλ_i)² / Σλ_i²    (effective dimensionality of the embedding cloud)

    Also returns eigenvalue spectrum (explained variance ratio, sorted desc).
    """
    Z_flat = Z.reshape(-1, Z.shape[-1])
    Z_c    = Z_flat - Z_flat.mean(0)
    cov    = (Z_c.T @ Z_c) / len(Z_c)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)[::-1]        # descending
    total  = eigvals.sum()
    pr     = total**2 / (eigvals**2).sum()
    evr    = eigvals / (total + 1e-12)
    return float(pr), evr


def msd_curve(Z, max_dt=None):
    """
    Mean Squared Displacement: MSD(Δt) = ⟨‖z_{t+Δt} − z_t‖₂²⟩_{n,t}

    Returns dt_vals, msd, (alpha, c) from power-law fit MSD ∝ c·Δt^α
    """
    T      = Z.shape[1]
    max_dt = min(max_dt or T // 2, T - 1)
    dt_arr = np.arange(1, max_dt + 1)
    msd    = np.zeros(max_dt)

    for i, dt in enumerate(dt_arr):
        diff   = Z[:, dt:, :] - Z[:, :T - dt, :]   # (N, T-dt, D)
        msd[i] = (diff**2).sum(-1).mean()

    # Power-law fit in log-log space
    log_dt  = np.log(dt_arr.astype(float))
    log_msd = np.log(msd + 1e-12)
    alpha, log_c, *_ = np.polyfit(log_dt, log_msd, 1, full=False)
    c = np.exp(log_c)

    return dt_arr, msd, float(alpha), float(c)


def geodesic_efficiency(Z):
    """
    Efficiency = chord_length / path_length ∈ (0, 1]

    efficiency = 1  ↔  straight line through latent space
    efficiency ≪ 1  ↔  winding, high-curvature trajectory

    Returns per-trajectory efficiencies: (N,)
    """
    chord = np.linalg.norm(Z[:, -1, :] - Z[:, 0, :], axis=-1)     # (N,)
    steps = np.linalg.norm(np.diff(Z, axis=1), axis=-1).sum(1)     # (N,)
    return chord / (steps + 1e-12)


def contact_curvature_analysis(kappa, states, thresh):
    """
    Correlate curvature with physical contact events.

    Contact proxy: the T-block only moves when the effector touches it, so
    block_velocity = ‖block_{t+1} − block_t‖₂ > thresh is a clean binary
    contact indicator without requiring exact agent-position parsing.

    State layout: [block_x, block_y, goal_x, goal_y, angle, ...]
    block_pos = state[:, :, 0:2]
    """
    block_pos    = states[:, :, 0:2]                              # (N, T, 2)
    block_vel    = np.linalg.norm(np.diff(block_pos, axis=1), axis=-1)  # (N, T-1)
    # Pad front so shape matches (N, T)
    block_vel    = np.concatenate([block_vel[:, :1], block_vel], axis=1)  # (N, T)
    dist         = block_vel                                      # repurpose for plotting
    contact      = block_vel > thresh                             # (N, T)

    # kappa has T-2 steps; align to frames [1 .. T-2] (centre of triplet)
    contact_k = contact[:, 1:-1]     # (N, T-2)

    k_contact    = kappa[contact_k]
    k_no_contact = kappa[~contact_k]

    # Mann-Whitney U: fast-push frames vs slow-push frames — which has sharper turns?
    stat, pval = sp_stats.mannwhitneyu(
        k_contact, k_no_contact, alternative='less'
    )

    # Spearman correlation: block speed vs curvature (continuous, no threshold)
    bv_k = dist[:, 1:-1].ravel()   # block_vel at curvature timestamps
    spearman_r, spearman_p = sp_stats.spearmanr(bv_k, kappa.ravel())

    return {
        'contact_mean_cos':    float(k_contact.mean()),
        'no_contact_mean_cos': float(k_no_contact.mean()),
        'contact_std_cos':     float(k_contact.std()),
        'no_contact_std_cos':  float(k_no_contact.std()),
        'contact_fraction':    float(contact_k.mean()),
        'mean_dist_pixels':    float(dist.mean()),
        'mannwhitney_U':       float(stat),
        'mannwhitney_pval':    float(pval),
        'spearman_r':          float(spearman_r),
        'spearman_p':          float(spearman_p),
        'dist_all':            dist,
        'contact_mask':        contact,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def fig_curvature_speed(kappa, speed, pr, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LeWM PushT — Latent Manifold Geometry (Vanilla Baseline)', fontsize=12)

    # ── (a) curvature histogram ───────────────────────────────────────────────
    ax = axes[0]
    flat = kappa.ravel()
    ax.hist(flat, bins=120, color='#4878D0', alpha=0.85, density=True)
    ax.axvline(flat.mean(),      color='crimson',    lw=2, label=f'mean = {flat.mean():.3f}')
    ax.axvline(np.median(flat),  color='darkorange', lw=2, linestyle='--',
               label=f'median = {np.median(flat):.3f}')
    ax.set_xlabel('cos(v_t , v_{t+1})  [curvature proxy]')
    ax.set_ylabel('Density')
    ax.set_title(f'(a) Curvature — σ = {flat.std():.3f}')
    ax.legend(fontsize=9)

    # ── (b) speed distribution ───────────────────────────────────────────────
    ax = axes[1]
    s_flat = speed.ravel()
    cv = s_flat.std() / (s_flat.mean() + 1e-12)
    ax.hist(s_flat, bins=100, color='#6ACC65', alpha=0.85, density=True)
    ax.axvline(s_flat.mean(), color='crimson', lw=2, label=f'mean = {s_flat.mean():.3f}')
    ax.set_xlabel('‖z_{t+1} − z_t‖₂  [latent speed]')
    ax.set_ylabel('Density')
    ax.set_title(f'(b) Latent Speed — CV = {cv:.3f}')
    ax.legend(fontsize=9)

    # ── (c) participation ratio annotation ───────────────────────────────────
    ax = axes[2]
    D  = 192
    ax.bar(['PR (used dims)', 'Total dims'], [pr, D],
           color=['#D65F5F', '#aaaaaa'], width=0.5)
    ax.set_ylabel('Dimensionality')
    ax.set_title(f'(c) Participation Ratio — {pr:.1f} / {D}')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = out_dir / 'fig1_curvature_speed_pr.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def fig_msd(dt_vals, msd, alpha, c, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dt_vals, msd, 'o', color='#4878D0', ms=4, label='MSD(Δt)')
    fit_line = c * dt_vals**alpha
    ax.loglog(dt_vals, fit_line, '--', color='crimson', lw=2,
              label=f'power-law fit  α = {alpha:.3f}')

    # Reference lines
    ax.loglog(dt_vals, c * dt_vals**1.0, ':', color='gray', lw=1, label='α=1 (ballistic)')
    ax.loglog(dt_vals, c * dt_vals**0.5, ':', color='silver', lw=1, label='α=0.5 (diffusion)')

    ax.set_xlabel('Δt  [steps]')
    ax.set_ylabel('‖z_{t+Δt} − z_t‖₂²  (MSD)')
    ax.set_title('(d) Temporal Metric Scaling Law\n'
                 'α=1 → straight manifold; α<1 → saturating / sub-ballistic')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / 'fig2_msd_scaling.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def fig_eigenspectrum(evr, pr, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    cumvar = np.cumsum(evr)
    dims   = np.arange(1, len(evr) + 1)

    ax.bar(dims[:80], evr[:80], color='#4878D0', alpha=0.8, label='Individual EVR')
    ax2 = ax.twinx()
    ax2.plot(dims[:80], cumvar[:80], color='crimson', lw=2, label='Cumulative')
    ax2.axhline(0.9, color='crimson', linestyle='--', lw=1, alpha=0.6, label='90% threshold')
    ax2.set_ylabel('Cumulative explained variance', color='crimson')
    ax2.tick_params(axis='y', colors='crimson')

    dims_90 = int(np.searchsorted(cumvar, 0.9)) + 1
    ax.set_xlabel('Principal component rank')
    ax.set_ylabel('Explained variance ratio')
    ax.set_title(f'(e) Embedding Eigenspectrum\n'
                 f'PR = {pr:.1f} / 192 dims     90% var in top {dims_90} dims')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    path = out_dir / 'fig3_eigenspectrum.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def fig_contact_correlation(kappa, contact_results, out_dir):
    contact_k    = kappa[contact_results['contact_mask'][:, 1:-1]]
    no_contact_k = kappa[~contact_results['contact_mask'][:, 1:-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('(f) Curvature at Contact vs Free-Space', fontsize=12)

    # ── density comparison ────────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(-1, 1, 80)
    ax.hist(contact_k,    bins=bins, density=True, alpha=0.7, color='crimson',
            label=f'contact  (n={len(contact_k):,})\nmean={contact_k.mean():.3f}')
    ax.hist(no_contact_k, bins=bins, density=True, alpha=0.5, color='#4878D0',
            label=f'free-space (n={len(no_contact_k):,})\nmean={no_contact_k.mean():.3f}')
    ax.set_xlabel('cos(v_t, v_{t+1})')
    ax.set_ylabel('Density')
    ax.set_title(f'p = {contact_results["mannwhitney_pval"]:.2e}  (Mann-Whitney U)')
    ax.legend(fontsize=9)

    # ── mean distance to block over trajectory (averaged across trajectories) ──
    ax = axes[1]
    dist = contact_results['dist_all']                   # (N, T)
    mean_dist = dist.mean(0)                             # (T,)
    std_dist  = dist.std(0)
    t_axis    = np.arange(len(mean_dist))
    ax.plot(t_axis, mean_dist, color='#4878D0', lw=2, label='mean block speed')
    ax.fill_between(t_axis, mean_dist - std_dist, mean_dist + std_dist,
                    alpha=0.2, color='#4878D0')
    ax.axhline(CONTACT_THRESH, color='crimson', linestyle='--', lw=1.5,
               label=f'contact threshold ({CONTACT_THRESH:.1f} px/step)')
    ax.set_xlabel('Step within trajectory window')
    ax.set_ylabel('‖block_{t+1} - block_t‖  [px/step]')
    ax.set_title('Block Velocity over Time\n(moves only when agent pushes it)')
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = out_dir / 'fig4_contact_correlation.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def fig_example_trajectories(Z, kappa, states, out_dir, n_show=4):
    """Plot curvature + speed timeseries for a handful of trajectories."""
    fig, axes = plt.subplots(n_show, 2, figsize=(13, 3 * n_show), sharex=True)
    fig.suptitle('(g) Per-trajectory Curvature & Latent Speed', fontsize=12)

    for i in range(n_show):
        k = kappa[i]          # (T-2,)
        s = np.linalg.norm(np.diff(Z[i], axis=0), axis=-1)   # (T-1,)
        t_k = np.arange(1, len(k) + 1)
        t_s = np.arange(len(s))

        block_pos  = states[i, :, 0:2]
        block_vel  = np.linalg.norm(np.diff(block_pos, axis=0), axis=-1)   # (T-1,)
        block_vel  = np.concatenate([[block_vel[0]], block_vel])             # (T,)
        in_contact = block_vel > CONTACT_THRESH

        ax_k, ax_s = axes[i, 0], axes[i, 1]

        ax_k.plot(t_k, k, lw=1.2, color='#4878D0')
        ax_k.axhline(0, color='gray', lw=0.8, linestyle='--')
        # Shade contact regions
        for t in range(len(in_contact) - 1):
            if in_contact[t]:
                ax_k.axvspan(t, t + 1, alpha=0.15, color='crimson')
        ax_k.set_ylabel('cos similarity')
        if i == 0:
            ax_k.set_title('Curvature (red=contact)')

        ax_s.plot(t_s, s, lw=1.2, color='#6ACC65')
        for t in range(len(in_contact) - 1):
            if in_contact[t]:
                ax_s.axvspan(t, t + 1, alpha=0.15, color='crimson')
        ax_s.set_ylabel('‖Δz‖₂')
        if i == 0:
            ax_s.set_title('Latent Speed')

    for ax in axes[-1]:
        ax.set_xlabel('Step')
    fig.tight_layout()
    path = out_dir / 'fig5_example_trajectories.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.img_size = IMG_SIZE    # encoder was trained at 56px
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== LeWM Latent Geometry Profiling ===')
    print(f'  ckpt     : {args.ckpt}')
    print(f'  num_traj : {args.num_traj}  ×  traj_len={args.traj_len}')
    print(f'  device   : {args.device}\n')

    # ── load model ────────────────────────────────────────────────────────────
    print('[1/6] Loading frozen encoder...')
    model = load_pretrained(args.ckpt)
    model.eval().requires_grad_(False)

    # ── extract embeddings ────────────────────────────────────────────────────
    print('[2/6] Extracting trajectory embeddings...')
    Z, states = extract_embeddings(model, args)
    print(f'  Z.shape = {Z.shape}   states.shape = {states.shape}')

    # ── curvature & speed ─────────────────────────────────────────────────────
    print('[3/6] Computing curvature and speed...')
    vel, kappa, speed = curvature_and_speed(Z)

    kappa_flat = kappa.ravel()
    speed_flat = speed.ravel()
    speed_cv   = speed_flat.std() / (speed_flat.mean() + 1e-12)
    geo_eff    = geodesic_efficiency(Z)

    print(f'  curvature — mean={kappa_flat.mean():.4f}  std={kappa_flat.std():.4f}  '
          f'median={np.median(kappa_flat):.4f}')
    print(f'  speed     — mean={speed_flat.mean():.4f}  CV={speed_cv:.4f}')
    print(f'  geodesic efficiency — mean={geo_eff.mean():.4f}  '
          f'std={geo_eff.std():.4f}  min={geo_eff.min():.4f}')

    # ── participation ratio ───────────────────────────────────────────────────
    print('[4/6] Computing participation ratio...')
    pr, evr = participation_ratio(Z)
    dims_90 = int(np.searchsorted(np.cumsum(evr), 0.9)) + 1
    print(f'  PR = {pr:.2f} / {Z.shape[-1]} dims   (90% var in top {dims_90} dims)')

    # ── MSD scaling law ───────────────────────────────────────────────────────
    print('[5/6] Computing MSD scaling law...')
    dt_vals, msd, alpha, c = msd_curve(Z)
    print(f'  MSD power law: MSD(Δt) ≈ {c:.4f} · Δt^{alpha:.4f}')
    if   alpha > 0.9:  regime = 'ballistic (linear) — well-structured manifold'
    elif alpha > 0.6:  regime = 'super-diffusive — partially structured'
    elif alpha > 0.4:  regime = 'diffusive (random walk!) — poor temporal structure'
    else:              regime = 'sub-diffusive — manifold compression / saturation'
    print(f'  regime: {regime}')

    # ── contact correlation ───────────────────────────────────────────────────
    print('[6/6] Contact correlation analysis...')
    contact_res = contact_curvature_analysis(kappa, states, args.contact_thresh)
    print(f'  contact fraction (block moving) : {contact_res["contact_fraction"]:.3f}')
    print(f'  mean block speed (px/step)      : {contact_res["mean_dist_pixels"]:.2f}')
    print(f'  curvature @ contact    : {contact_res["contact_mean_cos"]:.4f} '
          f'± {contact_res["contact_std_cos"]:.4f}')
    print(f'  curvature @ free-space : {contact_res["no_contact_mean_cos"]:.4f} '
          f'± {contact_res["no_contact_std_cos"]:.4f}')
    print(f'  Mann-Whitney U p-value : {contact_res["mannwhitney_pval"]:.3e}')
    print(f'  Spearman r(block_vel, curvature) : {contact_res["spearman_r"]:.4f}  '
          f'p={contact_res["spearman_p"]:.3e}')
    if contact_res['spearman_p'] < 0.05:
        sign = 'positive' if contact_res['spearman_r'] > 0 else 'negative'
        print(f'  → {sign} correlation: faster push → '
              + ('smoother' if contact_res['spearman_r'] > 0 else 'sharper') + ' latent turns')
    else:
        print('  → no significant correlation between block speed and latent curvature')

    # ── plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    fig_curvature_speed(kappa, speed, pr, args.out_dir)
    fig_msd(dt_vals, msd, alpha, c, args.out_dir)
    fig_eigenspectrum(evr, pr, args.out_dir)
    fig_contact_correlation(kappa, contact_res, args.out_dir)
    fig_example_trajectories(Z, kappa, states, args.out_dir)

    # ── save summary JSON ─────────────────────────────────────────────────────
    summary = {
        'ckpt': args.ckpt,
        'num_traj': args.num_traj,
        'traj_len': args.traj_len,
        'embed_dim': int(Z.shape[-1]),
        'curvature': {
            'mean':   float(kappa_flat.mean()),
            'std':    float(kappa_flat.std()),
            'median': float(np.median(kappa_flat)),
            'pct5':   float(np.percentile(kappa_flat, 5)),
            'pct95':  float(np.percentile(kappa_flat, 95)),
        },
        'speed': {
            'mean': float(speed_flat.mean()),
            'std':  float(speed_flat.std()),
            'cv':   float(speed_cv),
        },
        'geodesic_efficiency': {
            'mean': float(geo_eff.mean()),
            'std':  float(geo_eff.std()),
            'min':  float(geo_eff.min()),
        },
        'participation_ratio': float(pr),
        'dims_for_90pct_var':  int(dims_90),
        'msd_alpha':           float(alpha),
        'msd_c':               float(c),
        'msd_regime':          regime,
        'contact': {k: v for k, v in contact_res.items()
                    if not isinstance(v, np.ndarray)},
    }
    json_path = args.out_dir / 'geometry_summary.json'
    json_path.write_text(json.dumps(summary, indent=2))
    print(f'  saved {json_path}')

    # ── print research summary ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('LATENT GEOMETRY REPORT — LeWM PushT (epoch 10, 56px)')
    print('='*60)
    print(f'  Curvature (cos similarity)  mean={kappa_flat.mean():.3f}  '
          f'std={kappa_flat.std():.3f}')
    print(f'    → wide std means manifold has sharp turns')
    print(f'  Latent speed CV             {speed_cv:.3f}')
    print(f'    → CV>>0 means uneven temporal coverage')
    print(f'  Geodesic efficiency         {geo_eff.mean():.3f} ± {geo_eff.std():.3f}')
    print(f'    → 1.0=straight; <0.5=highly winding')
    print(f'  Participation ratio         {pr:.1f} / 192')
    print(f'    → 90% variance in top {dims_90} dims')
    print(f'  MSD power law α             {alpha:.3f}  [{regime}]')
    print(f'  Contact curvature p-value   {contact_res["mannwhitney_pval"]:.3e}')
    print('='*60)


if __name__ == '__main__':
    main()
