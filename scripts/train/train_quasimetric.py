"""Train a quasimetric distance head on top of a frozen LeWM encoder.

Learns d(z_i, z_j) ≈ step distance between observations i and j.
Positive pairs: same-trajectory (z_t, z_{t+k}), label = k model steps.
Negative pairs: cross-trajectory (z_t, z_{t'+k}), label = far (max_dist+margin).

Usage:
    uv run python scripts/train/train_quasimetric.py \
        --model lewm_agir_umaze/weights_epoch_0010.pt \
        --dataset ogb_pointmaze_umaze.lance \
        --output qm_agir_umaze \
        --epochs 20

Other envs pass their training resolution / frameskip, e.g. PushT:
    --dataset galilai-group/lewm-pusht --img-size 56 --frameskip 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.data import get_cache_dir
from stable_worldmodel.planning import QuasimetricHead
from stable_worldmodel.wm.utils import load_pretrained

BATCH_SIZE = 512
HIDDEN_DIM = 256
LR = 3e-4


def make_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


class PairDataset(Dataset):
    """Pairs of embeddings with step-distance labels (pre-computed)."""

    def __init__(self, z_pairs: np.ndarray, labels: np.ndarray):
        self.z_pairs = torch.from_numpy(z_pairs).float()  # (N, 2, D)
        self.labels = torch.from_numpy(labels).float()  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.z_pairs[idx, 0], self.z_pairs[idx, 1], self.labels[idx]


def encode_episode(
    model, transform, frames_hwc: np.ndarray, act_dim: int
) -> np.ndarray:
    """Encode (T, H, W, C) uint8 → (T, D) numpy embeddings."""
    ts = torch.stack([transform(f) for f in frames_hwc]).unsqueeze(
        0
    )  # (1,T,C,H,W)
    T = len(frames_hwc)
    with torch.no_grad():
        info = model.encode(
            {'pixels': ts, 'action': torch.zeros(1, T, act_dim)}
        )
    return info['emb'][0].numpy()  # (T, D)


def build_pairs(
    model,
    ds,
    ep_col: np.ndarray,
    step_col: np.ndarray,
    rng: np.random.Generator,
    transform,
    frameskip: int,
    max_dist: int,
    max_eps: int = 800,
    pairs_per_ep: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (z_i, z_j, label) training pairs from the dataset."""
    uniq_eps = np.unique(ep_col)
    selected = rng.choice(
        uniq_eps, size=min(max_eps, len(uniq_eps)), replace=False
    )

    all_pairs, all_labels = [], []

    # Cache per-episode embeddings
    ep_embs: dict[int, np.ndarray] = {}

    for ep_id in tqdm(selected, desc='Encoding episodes'):
        rows = np.where(ep_col == ep_id)[0]
        raw_steps = step_col[rows]
        if len(raw_steps) < frameskip * 3:
            continue
        s = int(raw_steps[0])
        n_model_steps = len(raw_steps) // frameskip
        span = n_model_steps * frameskip
        chunk = ds.load_chunk(
            np.array([ep_id]), np.array([s]), np.array([s + span])
        )[0]
        px = (
            chunk['pixels'][::frameskip].permute(0, 2, 3, 1).numpy()
        )  # (T, H,W,C)
        act_dim = chunk['action'].shape[-1] * frameskip
        embs = encode_episode(model, transform, px, act_dim)  # (T, D)
        ep_embs[ep_id] = embs

    # Positive pairs: same episode, distance = k model steps
    for ep_id, embs in ep_embs.items():
        T = len(embs)
        for _ in range(pairs_per_ep):
            i = int(rng.integers(0, T - 1))
            max_k = min(max_dist, T - 1 - i)
            if max_k < 1:
                continue
            k = int(rng.integers(1, max_k + 1))
            all_pairs.append([embs[i], embs[i + k]])
            all_labels.append(float(k))

    # Negative pairs: cross-episode, label = max_dist + 5
    ep_list = list(ep_embs.keys())
    n_neg = len(all_pairs) // 2
    for _ in range(n_neg):
        ep_a, ep_b = rng.choice(ep_list, size=2, replace=False)
        i = int(rng.integers(0, len(ep_embs[ep_a])))
        j = int(rng.integers(0, len(ep_embs[ep_b])))
        all_pairs.append([ep_embs[ep_a][i], ep_embs[ep_b][j]])
        all_labels.append(float(max_dist + 5))

    z_pairs = np.array(all_pairs, dtype=np.float32)  # (N, 2, D)
    labels = np.array(all_labels, dtype=np.float32)  # (N,)
    return z_pairs, labels


def train(args):
    rng = np.random.default_rng(42)
    device = 'cpu'

    print(f'Loading model: {args.model}')
    model = load_pretrained(args.model).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f'Loading dataset: {args.dataset}')
    ds = swm.data.load_dataset(args.dataset)
    ep_col = ds.get_col_data('episode_idx')
    step_col = ds.get_col_data('step_idx')

    print('Building training pairs…')
    neg_label = args.max_dist + 5
    z_pairs, labels = build_pairs(
        model,
        ds,
        ep_col,
        step_col,
        rng,
        transform=make_transform(args.img_size),
        frameskip=args.frameskip,
        max_dist=args.max_dist,
    )
    emb_dim = z_pairs.shape[-1]
    print(
        f'  {len(labels)} pairs  (pos: {(labels < neg_label).sum()}, neg: {(labels >= neg_label).sum()})'
    )

    # Normalise labels to [0, 1] range for stable training
    label_scale = float(neg_label)
    labels_norm = labels / label_scale

    # 90/10 split
    N = len(labels)
    idx = rng.permutation(N)
    n_train = int(0.9 * N)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    tr_ds = PairDataset(z_pairs[tr_idx], labels_norm[tr_idx])
    va_ds = PairDataset(z_pairs[va_idx], labels_norm[va_idx])
    tr_dl = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE)

    head = QuasimetricHead(emb_dim, HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    best_val, best_state = float('inf'), None
    for epoch in range(args.epochs):
        head.train()
        tr_loss = 0.0
        for z_i, z_j, lbl in tr_dl:
            z_i, z_j, lbl = z_i.to(device), z_j.to(device), lbl.to(device)
            pred = head(z_i, z_j)
            loss = F.huber_loss(pred, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        head.eval()
        va_loss = 0.0
        with torch.no_grad():
            for z_i, z_j, lbl in va_dl:
                z_i, z_j, lbl = z_i.to(device), z_j.to(device), lbl.to(device)
                va_loss += F.huber_loss(head(z_i, z_j), lbl).item()
        va_loss /= len(va_dl)
        tr_loss /= len(tr_dl)

        print(
            f'Epoch {epoch + 1:>3}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}'
        )
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Save
    ckpt_dir = Path(get_cache_dir(sub_folder='checkpoints')) / args.output
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'qm_head.pt'
    torch.save(
        {
            'state_dict': best_state,
            'emb_dim': emb_dim,
            'hidden_dim': HIDDEN_DIM,
            'label_scale': label_scale,
        },
        ckpt_path,
    )
    print(f'Saved to {ckpt_path}  (best_val={best_val:.4f})')

    # Quick sanity: Pearson r between predicted and true distance on val set
    head.load_state_dict(best_state)
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for z_i, z_j, lbl in va_dl:
            preds.append(head(z_i, z_j).numpy())
            trues.append(lbl.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    r = np.corrcoef(trues, preds)[0, 1]
    print(f'Val Pearson r(true_dist, pred_dist) = {r:.4f}')
    (ckpt_dir / 'qm_results.json').write_text(
        json.dumps({'best_val': best_val, 'pearson_r': r})
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='LeWM checkpoint name')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument(
        '--output', required=True, help='Output checkpoint name'
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument(
        '--img-size', type=int, default=64, help='training image resolution'
    )
    parser.add_argument(
        '--frameskip',
        type=int,
        default=2,
        help='training frameskip (raw steps per model step)',
    )
    parser.add_argument(
        '--max-dist',
        type=int,
        default=20,
        help='max positive-pair gap in model steps',
    )
    train(parser.parse_args())
