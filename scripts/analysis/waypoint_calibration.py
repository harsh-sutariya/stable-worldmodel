#!/usr/bin/env python3
"""
Dynamic Waypoint Extraction Calibration.

Converts AGIR latent trajectories into a Hierarchical Waypoint Dataset by
treating the latent velocity signal as a 1-D event detector.

Four-phase approach:
  Phase 1 — Signal characterization: distribution of V_smooth across the dataset.
  Phase 2 — Threshold sweep: run the state machine for 20 candidate τ_vel values,
             collect Δt histograms and compute bimodality statistics for each.
  Phase 3 — Goldilocks analysis: recommend the τ_vel that maximises the bimodality
             coefficient while avoiding failure modes (dt_max spike / dt_min stutter).
  Phase 4 — Dataset structure: show the macro-action distribution at τ_vel*.

State-machine parameters:
  τ_vel    — velocity threshold (to sweep and calibrate here)
  Δt_min   — debounce / NMS window (prevents waypoint stutter during contact)
  Δt_max   — free-space horizon cap  (guarantees L1 planner horizon is bounded)

Usage:
    uv run python scripts/analysis/waypoint_calibration.py \\
        --ckpt lewm_agir_pusht/weights_epoch_0010.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew, kurtosis

ROOT = Path('/Users/harsh/projects/personal/swm/stable-worldmodel')
sys.path.insert(0, str(ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained

# ── Dataset / model constants ─────────────────────────────────────────────────
DATASET    = 'galilai-group/lewm-pusht'
IMG_SIZE   = 56
FRAMESKIP  = 5

# ── Extraction budget ─────────────────────────────────────────────────────────
N_TRAJ     = 400       # number of trajectory windows to extract
TRAJ_LEN   = 45        # max frames with ≥1 sample (45 × frameskip=5 = 225 env steps ≈ 4.5 s)
BATCH_SIZE = 16

# ── Signal processing ─────────────────────────────────────────────────────────
SMOOTH_W   = 3         # causal MA window (no future leakage)

# ── State-machine fixed parameters ───────────────────────────────────────────
DT_MIN     = 3         # debounce: minimum steps between any two waypoints
DT_MAX     = 35        # free-space cap (< TRAJ_LEN-1=44 to allow at least one interior event)

# ── Threshold sweep ───────────────────────────────────────────────────────────
N_SWEEP    = 20        # number of τ_vel candidates
SWEEP_LO   = 5         # lower percentile of V_smooth to start sweep
SWEEP_HI   = 95        # upper percentile of V_smooth to end sweep

DEVICE     = 'mps'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def pixel_transform():
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source='pixels', target='pixels'),
        spt.data.transforms.Resize(IMG_SIZE, source='pixels', target='pixels'),
    )


@torch.no_grad()
def extract_latent_trajectories(model, device):
    """
    Encode N_TRAJ windows of TRAJ_LEN frames each.
    Returns a list of (TRAJ_LEN, D) numpy arrays.
    """
    dataset = swm.data.load_dataset(
        DATASET, num_steps=TRAJ_LEN, frameskip=FRAMESKIP,
        keys_to_load=['pixels'],
    )
    dataset.transform = pixel_transform()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
    )

    trajectories = []
    for batch in loader:
        pixels = batch['pixels'].to(device)          # (B, T, C, H, W)
        B, T = pixels.shape[:2]
        flat  = pixels.view(B * T, *pixels.shape[2:])
        cls   = model.encoder(flat).last_hidden_state[:, 0]
        emb   = model.projector(cls).view(B, T, -1).cpu().float().numpy()
        for b in range(B):
            trajectories.append(emb[b])
        if len(trajectories) >= N_TRAJ:
            break

    return trajectories[:N_TRAJ]


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def causal_ma(signal, w):
    """Causal moving average of window w — no future frame is ever read."""
    out = np.zeros_like(signal)
    for t in range(len(signal)):
        s = max(0, t - w + 1)
        out[t] = signal[s:t + 1].mean()
    return out


def compute_velocity_signals(trajectories):
    """
    For each (T, D) trajectory compute:
      V_raw    = ||z_{t+1} - z_t||_2  (T-1,)
      V_smooth = causal_ma(V_raw, SMOOTH_W)

    Returns lists of raw and smoothed velocity arrays.
    """
    raw_list, smooth_list = [], []
    for Z in trajectories:
        V_raw    = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        V_smooth = causal_ma(V_raw, SMOOTH_W)
        raw_list.append(V_raw)
        smooth_list.append(V_smooth)
    return raw_list, smooth_list


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

def extract_waypoints(V_smooth, tau_vel, dt_min=DT_MIN, dt_max=DT_MAX):
    """
    1-D event detector over a smoothed velocity signal.

    Condition A (dynamic trigger):
        V_smooth[t] > tau_vel  AND  t - last_wp >= dt_min
        → append t (contact / manipulation event)

    Condition B (horizon cap):
        t - last_wp >= dt_max  (regardless of velocity)
        → append t (free-space tick to bound prediction horizon)

    Always initialise W = [0]; always terminate with W.append(T).
    Returns (W, delta_ts).
    """
    T = len(V_smooth)
    W = [0]
    last_wp = 0

    for t in range(1, T):
        dt_since = t - last_wp

        if V_smooth[t] > tau_vel:
            if dt_since >= dt_min:           # Condition A with debounce
                W.append(t)
                last_wp = t
        else:
            if dt_since >= dt_max:           # Condition B: horizon cap
                W.append(t)
                last_wp = t

    if W[-1] != T:
        W.append(T)

    delta_ts = [W[i + 1] - W[i] for i in range(len(W) - 1)]
    return W, delta_ts


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def sarle_bimodality(x):
    """
    Sarle's bimodality coefficient.
      BC = (γ₁² + 1) / (κ + 3(n-1)²/((n-2)(n-3)))
    BC > 0.555 → evidence for bimodality.
    Pure uniform: BC = 5/9 ≈ 0.556. Pure normal: BC = 1/3.
    """
    n = len(x)
    if n < 4:
        return float('nan')
    g1  = skew(x)
    g2  = kurtosis(x, fisher=True)   # excess kurtosis
    num = g1 ** 2 + 1
    den = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-10)
    return float(num / (den + 1e-10))


def valley_analysis(dts, n_bins=None):
    """
    Detect a bimodal valley in the Δt histogram.

    Smooths counts with a width-3 uniform filter, locates the dominant
    left and right peaks, and measures the relative valley depth between them.

    Returns dict with keys: has_valley, valley_loc, relative_depth,
    left_peak_loc, right_peak_loc.
    """
    lo, hi = DT_MIN, DT_MAX + 1
    if n_bins is None:
        n_bins = hi - lo

    counts, edges = np.histogram(dts, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = uniform_filter1d(counts.astype(float), size=3)

    mid = len(smoothed) // 2
    lp  = int(np.argmax(smoothed[:mid]))
    rp  = mid + int(np.argmax(smoothed[mid:]))

    if lp >= rp:
        return dict(has_valley=False, valley_loc=float('nan'),
                    relative_depth=0.0, left_peak_loc=float('nan'),
                    right_peak_loc=float('nan'))

    segment  = smoothed[lp:rp + 1]
    valley_i = lp + int(np.argmin(segment))
    peak_h   = min(smoothed[lp], smoothed[rp])
    depth    = peak_h - smoothed[valley_i]
    rel_depth = float(depth / (peak_h + 1e-10))

    return dict(
        has_valley     = bool(rel_depth > 0.25),
        valley_loc     = float(centers[valley_i]),
        relative_depth = rel_depth,
        left_peak_loc  = float(centers[lp]),
        right_peak_loc = float(centers[rp]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────────────────────

def phase1_characterize(smooth_list):
    """Aggregate V_smooth statistics across the full dataset."""
    V_all = np.concatenate(smooth_list)
    pcts  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_v = np.percentile(V_all, pcts)
    return {
        'n_frames':    int(len(V_all)),
        'mean':        float(V_all.mean()),
        'std':         float(V_all.std()),
        'min':         float(V_all.min()),
        'max':         float(V_all.max()),
        'percentiles': {f'p{p}': float(v) for p, v in zip(pcts, pct_v)},
    }, V_all


def phase2_sweep(smooth_list, V_all):
    """
    Sweep τ_vel from p{SWEEP_LO} to p{SWEEP_HI} of V_smooth.
    For each candidate, run the state machine on all trajectories
    and compute Δt statistics and bimodality metrics.
    """
    lo = float(np.percentile(V_all, SWEEP_LO))
    hi = float(np.percentile(V_all, SWEEP_HI))
    thresholds = np.linspace(lo, hi, N_SWEEP)

    rows = []
    for tau in thresholds:
        all_dts = []
        total_wps = 0
        for V in smooth_list:
            _, dts = extract_waypoints(V, tau_vel=tau)
            all_dts.extend(dts)
            total_wps += len(dts) + 1

        dts_arr = np.array(all_dts, dtype=float)
        bc      = sarle_bimodality(dts_arr)
        valley  = valley_analysis(dts_arr)

        rows.append({
            'tau_vel':              float(tau),
            'mean_dt':              float(dts_arr.mean()),
            'median_dt':            float(np.median(dts_arr)),
            'std_dt':               float(dts_arr.std()),
            'p25_dt':               float(np.percentile(dts_arr, 25)),
            'p75_dt':               float(np.percentile(dts_arr, 75)),
            'bimodality_coeff':     float(bc),
            'frac_at_dtmax':        float((dts_arr >= DT_MAX).mean()),
            'frac_at_dtmin':        float((dts_arr <= DT_MIN).mean()),
            'n_waypoints_per_traj': float(total_wps / len(smooth_list)),
            'n_macro_actions':      int(len(dts_arr)),
            'valley':               valley,
            '_dts':                 list(dts_arr.astype(int)),   # kept for phase 3 only
        })

    return rows


def phase3_recommend(rows):
    """
    Select τ_vel* that maximises bimodality coefficient subject to:
      - frac_at_dtmax < 0.35   (not dominated by the free-space cap)
      - frac_at_dtmin < 0.35   (not stuttering at every contact frame)
      - mean_dt in (DT_MIN*2, DT_MAX*0.8)   (sane macro-action length)
    Fall back to pure BC maximum if no candidate passes all filters.
    """
    valid = [
        r for r in rows
        if r['frac_at_dtmax'] < 0.35
        and r['frac_at_dtmin'] < 0.35
        and DT_MIN * 2 < r['mean_dt'] < DT_MAX * 0.8
    ]
    pool = valid if valid else rows
    return max(pool, key=lambda r: r['bimodality_coeff'])


def phase4_dataset_structure(trajectories, smooth_list, tau_star):
    """
    At the recommended threshold, report macro-action distribution and
    print a concrete example segmentation.
    """
    all_dts = []
    wp_counts = []
    for V in smooth_list:
        _, dts = extract_waypoints(V, tau_vel=tau_star)
        all_dts.extend(dts)
        wp_counts.append(len(dts) + 1)

    dts_arr = np.array(all_dts, dtype=float)

    # Example segmentation on the first trajectory
    Z_ex  = trajectories[0]
    V_ex  = smooth_list[0]
    W_ex, dts_ex = extract_waypoints(V_ex, tau_vel=tau_star)

    return {
        'tau_star':              float(tau_star),
        'total_macro_actions':   int(len(dts_arr)),
        'mean_waypoints_per_traj': float(np.mean(wp_counts)),
        'mean_dt':               float(dts_arr.mean()),
        'std_dt':                float(dts_arr.std()),
        'p25_dt':                float(np.percentile(dts_arr, 25)),
        'p50_dt':                float(np.median(dts_arr)),
        'p75_dt':                float(np.percentile(dts_arr, 75)),
        'frac_short_le10':       float((dts_arr <= 10).mean()),
        'frac_mid_10_20':        float(((dts_arr > 10) & (dts_arr < 20)).mean()),
        'frac_long_ge20':        float((dts_arr >= 20).mean()),
        'example': {
            'T': int(len(Z_ex)),
            'W': [int(w) for w in W_ex],
            'delta_ts': [int(d) for d in dts_ex],
        },
        '_all_dts': [int(d) for d in dts_arr],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASCII helpers
# ─────────────────────────────────────────────────────────────────────────────

def ascii_histogram(values, n_bins=20, width=50, lo=None, hi=None, label=''):
    lo = lo or min(values)
    hi = hi or max(values)
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    mx = max(counts) if max(counts) > 0 else 1
    if label:
        print(f'  {label}')
    for i in range(len(counts)):
        bar = '█' * int(width * counts[i] / mx)
        print(f'  {edges[i]:6.1f} ─ {edges[i+1]:6.1f} │{bar:<{width}}│ {counts[i]}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--device', default=DEVICE)
    ap.add_argument('--out',    type=Path,
                    default=ROOT / 'outputs' / 'diagnostics' / 'waypoint_calibration.json')
    args = ap.parse_args()

    SEP  = '═' * 72
    sep2 = '─' * 72

    print(f'\n{SEP}')
    print(f'  DYNAMIC WAYPOINT EXTRACTION — CALIBRATION STUDY')
    print(f'  Model  : {args.ckpt}')
    print(f'  Config : smooth_w={SMOOTH_W}  dt_min={DT_MIN}  dt_max={DT_MAX}')
    print(f'  Budget : {N_TRAJ} trajectories × {TRAJ_LEN} frames (frameskip={FRAMESKIP})')
    print(f'{SEP}\n')

    # ── Load model ─────────────────────────────────────────────────────────
    print('Loading model …')
    model = load_pretrained(args.ckpt)
    model.to(args.device).eval()
    print('  OK\n')

    # ── Extract latent trajectories ────────────────────────────────────────
    print('Extracting latent trajectories …')
    trajectories = extract_latent_trajectories(model, args.device)
    D = trajectories[0].shape[1]
    print(f'  {len(trajectories)} trajectories  ×  {TRAJ_LEN} frames  ×  D={D}\n')

    # ── Compute velocity signals ───────────────────────────────────────────
    print(f'Computing V_smooth (causal MA w={SMOOTH_W}) …')
    _, smooth_list = compute_velocity_signals(trajectories)
    print('  Done\n')

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1
    # ═══════════════════════════════════════════════════════════════════════
    print(f'{sep2}')
    print('PHASE 1 — Signal Characterization')
    print(f'{sep2}')
    p1, V_all = phase1_characterize(smooth_list)

    print(f'  Frames analysed : {p1["n_frames"]:,}')
    print(f'  Mean  V_smooth  : {p1["mean"]:.5f}')
    print(f'  Std   V_smooth  : {p1["std"]:.5f}')
    print(f'  Range           : [{p1["min"]:.5f}, {p1["max"]:.5f}]')
    print()
    print('  Percentile map:')
    for k, v in p1['percentiles'].items():
        print(f'    {k:>4} : {v:.5f}')

    print()
    print('  V_smooth distribution (all frames):')
    ascii_histogram(V_all, n_bins=20, width=48, lo=0, hi=p1['percentiles']['p99'])

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2
    # ═══════════════════════════════════════════════════════════════════════
    print(f'\n{sep2}')
    print('PHASE 2 — Threshold Sweep')
    print(f'{sep2}')
    print(f'  Sweeping {N_SWEEP} values of τ_vel in '
          f'[p{SWEEP_LO}={p1["percentiles"][f"p{SWEEP_LO}"]:.4f}, '
          f'p{SWEEP_HI}={p1["percentiles"][f"p{SWEEP_HI}"]:.4f}]\n')

    hdr = (f'  {"τ_vel":>9} │ {"mean Δt":>8} │ {"std Δt":>7} │ '
           f'{"BC":>6} │ {"@max":>6} │ {"@min":>6} │ {"WP/tr":>6} │ valley')
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))

    rows = phase2_sweep(smooth_list, V_all)
    for r in rows:
        vmark = '✓' if r['valley']['has_valley'] else ' '
        bc_flag = '*' if r['bimodality_coeff'] > 0.555 else ' '
        print(f'  {r["tau_vel"]:>9.4f} │ {r["mean_dt"]:>8.2f} │ {r["std_dt"]:>7.2f} │ '
              f'{r["bimodality_coeff"]:>5.3f}{bc_flag} │ {r["frac_at_dtmax"]:>6.3f} │ '
              f'{r["frac_at_dtmin"]:>6.3f} │ {r["n_waypoints_per_traj"]:>6.1f} │ {vmark}')

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3
    # ═══════════════════════════════════════════════════════════════════════
    print(f'\n{sep2}')
    print('PHASE 3 — Goldilocks Analysis → Recommended τ_vel*')
    print(f'{sep2}')

    best = phase3_recommend(rows)
    tau_star = best['tau_vel']
    bc_str   = f'{best["bimodality_coeff"]:.4f}  ({"BIMODAL ✓" if best["bimodality_coeff"] > 0.555 else "unimodal"})'

    print(f'\n  Recommended   τ_vel*  =  {tau_star:.6f}')
    print(f'  Bimodality coefficient :  {bc_str}')
    print(f'  Valley detected        :  {best["valley"]["has_valley"]}')
    if best['valley']['has_valley']:
        v = best['valley']
        print(f'    Left peak  @ Δt ≈ {v["left_peak_loc"]:.1f}   (manipulation / contact)')
        print(f'    Valley     @ Δt ≈ {v["valley_loc"]:.1f}')
        print(f'    Right peak @ Δt ≈ {v["right_peak_loc"]:.1f}   (free-space approach)')
        print(f'    Relative depth = {v["relative_depth"]:.3f}')
    print()
    print(f'  Δt statistics at τ_vel*:')
    print(f'    mean   = {best["mean_dt"]:.2f}')
    print(f'    std    = {best["std_dt"]:.2f}')
    print(f'    IQR    = [{best["p25_dt"]:.1f}, {best["p75_dt"]:.1f}]')
    print(f'    @dtmax = {best["frac_at_dtmax"]:.3f}  '
          f'({"OK" if best["frac_at_dtmax"] < 0.20 else "WARNING: high dt_max pin"})')
    print(f'    @dtmin = {best["frac_at_dtmin"]:.3f}  '
          f'({"OK" if best["frac_at_dtmin"] < 0.20 else "WARNING: stutter risk"})')

    print()
    print(f'  Δt histogram at τ_vel*:')
    ascii_histogram(best['_dts'], n_bins=DT_MAX - DT_MIN + 1,
                    width=48, lo=DT_MIN, hi=DT_MAX + 1,
                    label='(each bar = 1 Δt bin)')

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4
    # ═══════════════════════════════════════════════════════════════════════
    print(f'\n{sep2}')
    print('PHASE 4 — Dataset Structure at τ_vel*')
    print(f'{sep2}')

    p4 = phase4_dataset_structure(trajectories, smooth_list, tau_star)

    print(f'  Total macro-actions         : {p4["total_macro_actions"]:,}')
    print(f'  Mean waypoints per trajectory : {p4["mean_waypoints_per_traj"]:.1f}')
    print(f'  Δt mean / std               : {p4["mean_dt"]:.2f} / {p4["std_dt"]:.2f}')
    print(f'  Δt percentiles [25,50,75]   : [{p4["p25_dt"]:.1f}, {p4["p50_dt"]:.1f}, {p4["p75_dt"]:.1f}]')
    print()
    print('  Macro-action phase split:')
    print(f'    Short  (Δt ≤ 10) — manipulation / contact : {p4["frac_short_le10"]:.1%}')
    print(f'    Mid    (10 < Δt < 20)                     : {p4["frac_mid_10_20"]:.1%}')
    print(f'    Long   (Δt ≥ 20) — free-space approach    : {p4["frac_long_ge20"]:.1%}')
    print()
    ex = p4['example']
    print(f'  Example segmentation (trajectory 0, T={ex["T"]}):')
    print(f'    W        = {ex["W"]}')
    print(f'    delta_ts = {ex["delta_ts"]}')

    # ── Save ───────────────────────────────────────────────────────────────
    payload = {
        'model_ckpt': args.ckpt,
        'config': {
            'smooth_w': SMOOTH_W,
            'dt_min':   DT_MIN,
            'dt_max':   DT_MAX,
            'n_traj':   len(trajectories),
            'traj_len': TRAJ_LEN,
        },
        'phase1': p1,
        'phase2_sweep': [
            {k: v for k, v in r.items() if k != '_dts'}
            for r in rows
        ],
        'recommended': {k: v for k, v in best.items() if k != '_dts'},
        'phase4': {k: v for k, v in p4.items() if k != '_all_dts'},
        'delta_t_sample': p4['_all_dts'][:2000],   # first 2 k for visualisation
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'\n  Saved → {args.out}')

    print(f'\n{SEP}')
    print(f'  RECOMMENDATION')
    print(f'  τ_vel*  = {tau_star:.6f}')
    print(f'  Δt_min  = {DT_MIN}   (debounce)')
    print(f'  Δt_max  = {DT_MAX}   (free-space cap)')
    print(f'{SEP}\n')


if __name__ == '__main__':
    main()
