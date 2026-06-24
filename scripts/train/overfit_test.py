"""
Overfit test for LeWM.

Uses one fixed synthetic batch as train, val, and test (true overfit scenario).
Verifies:
  1. Model instantiates without errors
  2. Forward pass produces finite outputs
  3. Every parameter receives a non-zero gradient after the first backward
  4. Prediction loss drops to < 10% of its initial value in N_STEPS

No dataset download required. Run from the repo root with:
    python scripts/train/overfit_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
from loguru import logger as logging

import stable_pretraining as spt
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

# ── hyperparams ───────────────────────────────────────────────────────────────
IMG_SIZE = 56      # 56/14 = 4 → 4×4 = 16 patches; small enough to be fast on CPU
PATCH_SIZE = 14
EMBED_DIM = 64     # projector output dim (kept small for speed)
HISTORY_SIZE = 3   # matches lewm.yaml wm.history_size
NUM_PREDS = 1      # matches lewm.yaml wm.num_preds
ACTION_DIM = 2     # PushT action space
N_STEPS = 100
LR = 1e-3
CONVERGENCE_RATIO = 0.10   # final loss must be < 10% of initial to pass
# ─────────────────────────────────────────────────────────────────────────────


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _make_batch(device: torch.device) -> dict:
    """Single fixed batch used identically for every training step."""
    torch.manual_seed(42)
    T = HISTORY_SIZE + NUM_PREDS   # 4
    return {
        'pixels': torch.rand(1, T, 3, IMG_SIZE, IMG_SIZE, device=device),
        'action': torch.rand(1, T, ACTION_DIM, device=device),
    }


def _build_model(device: torch.device) -> LeWM:
    encoder = spt.backbone.utils.vit_hf(
        size='tiny',
        patch_size=PATCH_SIZE,
        image_size=IMG_SIZE,
        pretrained=False,
        use_mask_token=False,
    )
    vit_dim = encoder.config.hidden_size  # 192 for ViT-tiny

    model = LeWM(
        encoder=encoder,
        predictor=Predictor(
            num_frames=HISTORY_SIZE,
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM,
            output_dim=EMBED_DIM,
            depth=2,
            heads=4,
            mlp_dim=EMBED_DIM * 2,
            dim_head=16,
        ),
        action_encoder=Embedder(input_dim=ACTION_DIM, emb_dim=EMBED_DIM),
        projector=MLP(
            input_dim=vit_dim,
            hidden_dim=EMBED_DIM * 2,
            output_dim=EMBED_DIM,
        ),
        pred_proj=MLP(
            input_dim=EMBED_DIM,
            hidden_dim=EMBED_DIM * 2,
            output_dim=EMBED_DIM,
        ),
    ).to(device)
    return model


def _forward(model: LeWM, batch: dict) -> torch.Tensor:
    """Replicates lejepa_forward from lewm.py exactly."""
    output = model.encode(batch)
    emb = output['emb']           # (B, T, D)
    act_emb = output['act_emb']   # (B, T, D)

    ctx_emb = emb[:, :HISTORY_SIZE]       # (B, 3, D)
    ctx_act = act_emb[:, :HISTORY_SIZE]   # (B, 3, D)
    tgt_emb = emb[:, NUM_PREDS:]          # (B, 3, D)  ← emb at t+1..T

    pred_emb = model.predict(ctx_emb, ctx_act)   # (B, 3, D)
    return (pred_emb - tgt_emb).pow(2).mean()


def _is_adaln_gated(name: str) -> bool:
    """Parameters whose gradient is legitimately zero at step 0 due to AdaLN-zero init.

    ConditionalBlock initialises adaLN_modulation with W=0, b=0.
    Consequence:
      - gate_msa = gate_mlp = 0  →  grad zeroed for .attn.* and .mlp.*
      - dOut/d(action_emb) = W^T = 0  →  grad zeroed for action_encoder.*
    After the first optimizer step W becomes non-zero and all these params
    receive proper gradients.
    """
    if 'predictor.transformer.layers' in name and ('.attn.' in name or '.mlp.' in name):
        return True
    if name.startswith('action_encoder.'):
        return True
    return False


def _check_gradients(model: LeWM, step: int) -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_dead_param_names).

    On step 0 we exempt AdaLN-gated params (expected zero due to init).
    From step 1 onward every param must have a non-zero gradient.
    """
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad
        and (p.grad is None or p.grad.abs().sum() == 0)
        and not (step == 0 and _is_adaln_gated(name))
    ]
    return len(dead) == 0, dead


def main():
    device = _pick_device()
    logging.info(f'Device        : {device}')
    logging.info(f'Image size    : {IMG_SIZE}×{IMG_SIZE}  ({(IMG_SIZE//PATCH_SIZE)**2} patches)')

    batch = _make_batch(device)
    model = _build_model(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Parameters    : {n_params:,}')
    logging.info(f'Steps         : {N_STEPS}  (same batch every step)')
    logging.info('')

    opt = optim.Adam(model.parameters(), lr=LR)
    losses: list[float] = []

    for step in range(N_STEPS):
        opt.zero_grad()
        loss = _forward(model, batch)

        assert torch.isfinite(loss), f'step {step}: loss is {loss.item()} (NaN or Inf)'
        loss.backward()

        # Gradient check: step 0 exempts AdaLN-gated params (gate=0 by init);
        # step 1 checks everything (gates have opened after first update).
        if step in (0, 1):
            ok, dead = _check_gradients(model, step)
            if not ok:
                raise AssertionError(
                    f'step {step}: {len(dead)} parameter(s) have zero/missing gradient:\n'
                    + '\n'.join(f'  {n}' for n in dead[:10])
                )
            if step == 0:
                logging.info(
                    f'Gradient check (step 0): PASSED '
                    f'— encoder/projector/action_encoder/pred_proj/adaLN all have grads; '
                    f'predictor attn/mlp zero by AdaLN-zero init (expected)'
                )
            else:
                logging.info(f'Gradient check (step 1): PASSED — all {n_params:,} params have grads after gate open')
                logging.info('')

        opt.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == N_STEPS - 1:
            logging.info(f'step {step:3d}  loss = {loss.item():.6f}')

    # ── final assertions ──────────────────────────────────────────────────────
    ratio = losses[-1] / (losses[0] + 1e-9)
    converged = ratio < CONVERGENCE_RATIO

    logging.info('')
    logging.info(f'Initial loss  : {losses[0]:.6f}')
    logging.info(f'Final loss    : {losses[-1]:.6f}  (×{ratio:.4f} of initial)')

    assert converged, (
        f'Loss did not converge: {losses[0]:.6f} → {losses[-1]:.6f} '
        f'(ratio {ratio:.3f} >= {CONVERGENCE_RATIO})'
    )

    logging.info(f'Convergence   : PASSED (ratio {ratio:.4f} < {CONVERGENCE_RATIO})')
    logging.info('')
    logging.info('══════  Overfit test PASSED  ══════')


if __name__ == '__main__':
    main()
