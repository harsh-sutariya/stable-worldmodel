"""Tests for the pluggable planning objectives.

Direct-behavior tests for the concrete
:class:`~stable_worldmodel.protocols.Objective` implementations in
:mod:`stable_worldmodel.planning.objective` (GoalMSE / ControlPenalty /
QuasimetricCost / WeightedSum). Their use inside the ``ShootingCostEvaluator`` seam is covered in
``test_evaluator.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from stable_worldmodel.planning import (
    ControlPenalty,
    GoalMSE,
    QuasimetricCost,
    QuasimetricHead,
    WeightedSum,
)

# CEM-like dimensions.
B, S, T, D, H, A = 2, 3, 2, 5, 2, 4


def test_goal_mse_formula():
    pred = torch.randn(B, S, T, D)
    goal = torch.randn(B, T, D)
    out = GoalMSE()({'predicted_emb': pred, 'goal_emb': goal})

    goal_b = goal[:, None, -1:, :].expand_as(pred)
    expected = F.mse_loss(
        pred[..., -1:, :], goal_b[..., -1:, :], reduction='none'
    ).sum(dim=tuple(range(2, pred.ndim)))
    torch.testing.assert_close(out, expected)


def test_control_penalty_formula():
    ac = torch.randn(B, S, H, A)
    out = ControlPenalty()({'action_candidates': ac})
    torch.testing.assert_close(out, ac.pow(2).sum(dim=(2, 3)))


def test_weighted_sum_combines_terms():
    ac = torch.randn(B, S, H, A)
    info = {'action_candidates': ac}
    combined = WeightedSum([(2.0, ControlPenalty()), (0.5, ControlPenalty())])
    torch.testing.assert_close(combined(info), 2.5 * ControlPenalty()(info))


def test_weighted_sum_requires_terms():
    with pytest.raises(ValueError):
        WeightedSum([])


@pytest.fixture
def qm_ckpt(tmp_path):
    torch.manual_seed(0)
    head = QuasimetricHead(D, hidden_dim=8)
    path = tmp_path / 'qm_head.pt'
    torch.save(
        {
            'state_dict': head.state_dict(),
            'emb_dim': D,
            'hidden_dim': 8,
        },
        path,
    )
    return head, path


def test_quasimetric_cost_scores_last_step_against_goal(qm_ckpt):
    head, path = qm_ckpt
    pred = torch.randn(B, S, T, D)
    goal = torch.randn(B, T, D)
    out = QuasimetricCost(str(path))({'predicted_emb': pred, 'goal_emb': goal})

    assert out.shape == (B, S)
    for b in range(B):
        for s in range(S):
            expected = head(pred[b, s, -1][None], goal[b, -1][None])
            torch.testing.assert_close(out[b, s], expected[0])


def test_quasimetric_cost_resolves_relative_path_under_cache(
    qm_ckpt, tmp_path, monkeypatch
):
    _, path = qm_ckpt
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    (tmp_path / 'checkpoints' / 'run').mkdir(parents=True)
    path.rename(tmp_path / 'checkpoints' / 'run' / 'qm_head.pt')

    cost = QuasimetricCost('run/qm_head.pt')
    out = cost(
        {
            'predicted_emb': torch.randn(B, S, T, D),
            'goal_emb': torch.randn(B, T, D),
        }
    )
    assert out.shape == (B, S)


def test_quasimetric_cost_head_is_frozen(qm_ckpt):
    _, path = qm_ckpt
    cost = QuasimetricCost(str(path))
    assert not any(p.requires_grad for p in cost.parameters())
