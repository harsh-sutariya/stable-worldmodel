"""Pluggable planning objectives, decoupled from the world model.

An :class:`~stable_worldmodel.protocols.Objective` scores a populated
``info_dict`` (after the world model has rolled out candidate actions) and
returns a per-candidate cost tensor of shape ``(B, S)``. Objectives are plain,
composable objects: swapping or combining costs no longer requires subclassing
the world model. The ``Objective`` protocol itself lives in
:mod:`stable_worldmodel.protocols`; this module holds concrete objectives.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.data import get_cache_dir
from stable_worldmodel.protocols import Objective


class QuasimetricHead(nn.Module):
    """MLP regressing the step distance between two latent embeddings.

    Shared by ``scripts/train/train_quasimetric.py`` (training) and
    :class:`QuasimetricCost` (planning) so the two cannot drift apart.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self, z_curr: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        """``z_curr``, ``z_goal``: ``(N, D)`` -> distances ``(N,)``."""
        return self.net(torch.cat([z_curr, z_goal], dim=-1)).squeeze(-1)


class GoalMSE(nn.Module):
    """Last-step MSE between predicted and goal embeddings.

    Reads ``predicted_emb`` ``(B, S, T, ...)`` and ``goal_emb``
    ``(B, T_goal, ...)``; returns per-candidate cost ``(B, S)``. The comparison
    is against the *last* predicted step and the *last* goal frame.

    Rank-agnostic: the time axis is addressed positionally (dim 2 of the
    prediction, dim 1 of the goal) rather than from the right, so latents that
    carry extra trailing axes work too — ``PreJEPA``'s pixel embedding has a
    patch axis, ``(B, S, T, patches, dim)``, and indexing from the right would
    silently slice patches instead of time.

    Point ``pred_key``/``goal_key`` at one source of a split latent (e.g.
    ``predicted_pixels_emb`` / ``pixels_goal_emb``) and combine the terms with
    :class:`WeightedSum` to score models whose latent concatenates several
    sources; see the planning guide.

    Behavior-preserving extraction of ``LeWM.criterion`` (reproduces it
    bit-for-bit), so an existing model migrates to the ``ShootingCostEvaluator`` seam
    without changing results.

    Args:
        pred_key: ``info_dict`` key holding the rollout predictions.
        goal_key: ``info_dict`` key holding the goal embedding.
        reduction: How the squared error is collapsed over every axis after
            the candidate axis. ``'sum'`` (the default) matches
            ``LeWM.criterion``; ``'mean'`` matches the former
            ``PreJEPA.criterion``. This choice also sets the *relative* weight
            of sources with different shapes, so it matters when composing
            per-source terms: summing over a 196x384 pixel embedding and a
            10-dim proprio embedding weights them ~7500x apart, whereas
            averaging puts them on a comparable scale.
    """

    def __init__(
        self,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
        reduction: str = 'sum',
    ) -> None:
        super().__init__()
        if reduction not in ('sum', 'mean'):
            raise ValueError(
                f"reduction must be 'sum' or 'mean', got {reduction!r}"
            )
        self.pred_key = pred_key
        self.goal_key = goal_key
        self.reduction = reduction

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]
        goal_emb = info_dict[self.goal_key][:, None, -1:].expand_as(pred_emb)
        err = F.mse_loss(
            pred_emb[:, :, -1:],
            goal_emb[:, :, -1:].detach(),
            reduction='none',
        )
        dims = tuple(range(2, pred_emb.ndim))
        return (
            err.sum(dim=dims)
            if self.reduction == 'sum'
            else err.mean(dim=dims)
        )


class ControlPenalty(nn.Module):
    """L2 penalty on the action candidates themselves.

    Reads ``action_candidates`` (shape ``(B, S, H, action_dim)``) from the
    ``info_dict`` — the ``ShootingCostEvaluator`` stores them there before scoring —
    and returns a per-candidate cost ``(B, S)``. Demonstrates a cost the world
    model never had to know about.
    """

    def __init__(self, action_key: str = 'action_candidates') -> None:
        super().__init__()
        self.action_key = action_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        actions = info_dict[self.action_key]
        return actions.pow(2).sum(dim=tuple(range(2, actions.ndim)))


class QuasimetricCost(nn.Module):
    """Learned distance to the goal as the planning cost (TD-JEPA style).

    Scores each candidate with ``d(z_pred, z_goal)`` from a
    :class:`QuasimetricHead` trained on frozen embeddings of the *same* world
    model (see ``scripts/train/train_quasimetric.py``). Same interface as
    :class:`GoalMSE`: reads ``predicted_emb`` ``(B, S, T, D)`` and
    ``goal_emb`` ``(B, T_goal, D)``, returns ``(B, S)``.

    Raw L2 (:class:`GoalMSE`) remains the default objective; this one is
    opt-in and helps on route-finding tasks (mazes, TwoRoom) but can hurt on
    contact-rich manipulation (PushT, Cube).

    Args:
        ckpt_path: ``qm_head.pt`` written by the trainer. A relative path is
            resolved under the checkpoints cache directory, like ``policy``.
        pred_key: ``info_dict`` key holding the rollout predictions.
        goal_key: ``info_dict`` key holding the goal embedding.
    """

    def __init__(
        self,
        ckpt_path: str,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.goal_key = goal_key

        path = Path(ckpt_path).expanduser()
        if not path.is_absolute() and not path.exists():
            path = get_cache_dir(sub_folder='checkpoints') / path
        ckpt = torch.load(path, map_location='cpu', weights_only=True)

        self.head = QuasimetricHead(ckpt['emb_dim'], ckpt['hidden_dim'])
        self.head.load_state_dict(ckpt['state_dict'])
        self.head.requires_grad_(False)

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict[self.pred_key]  # (B, S, T, D)
        goal_emb = info_dict[self.goal_key]  # (B, T_goal, D)

        # Planning moves the world model to the accelerator after this
        # objective is built, so follow the inputs' device lazily.
        if next(self.head.parameters()).device != pred_emb.device:
            self.head.to(pred_emb.device)

        B, S = pred_emb.shape[:2]
        z_pred = pred_emb[:, :, -1].reshape(B * S, -1)  # last predicted step
        z_goal = goal_emb[:, None, -1].expand(B, S, -1).reshape(B * S, -1)

        with torch.no_grad():
            return self.head(z_pred, z_goal).reshape(B, S)


class WeightedSum(nn.Module):
    """Linear combination of objectives: ``sum(w * term(info_dict))``.

    Lets us assemble multi-term costs (goal distance + control penalty +
    constraints) without touching the model or the solver.
    """

    def __init__(self, terms: list[tuple[float, Objective]]) -> None:
        super().__init__()
        if not terms:
            raise ValueError('WeightedSum requires at least one term')
        self.weights = [weight for weight, _ in terms]
        self.objectives = nn.ModuleList(term for _, term in terms)

    def forward(self, info_dict: dict) -> torch.Tensor:
        total = None
        for weight, term in zip(self.weights, self.objectives):
            value = weight * term(info_dict)
            total = value if total is None else total + value
        return total
