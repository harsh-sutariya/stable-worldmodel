"""Collect PointMaze expert demonstrations using BFS oracle navigation.

Usage:
    uv run python scripts/data/collect_pointmaze.py
"""

from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy

DATASET_PATH = (
    Path(swm.data.utils.get_cache_dir())
    / 'datasets'
    / 'ogb_pointmaze_large.lance'
)

NUM_ENVS = 8
NUM_EPISODES = 3000
MAX_EPISODE_STEPS = 250
IMAGE_SHAPE = (64, 64)


class PointMazeOraclePolicy(BasePolicy):
    """BFS-guided oracle policy for PointMaze.

    Uses the OGBench ``get_oracle_subgoal`` BFS to find the next waypoint
    toward the current episode goal, then applies proportional control.
    """

    def get_action(self, info_dict):
        n_envs = len(self.env.envs)
        actions = np.zeros((n_envs, 2), dtype=np.float32)

        for i, env_wrapper in enumerate(self.env.envs):
            base_env = env_wrapper.unwrapped
            cur_xy = np.array(base_env.get_xy(), dtype=np.float64)
            goal_xy = np.array(base_env.cur_task_info['goal_xy'], dtype=np.float64)
            subgoal_xy, _ = base_env.get_oracle_subgoal(cur_xy, goal_xy)
            direction = np.array(subgoal_xy, dtype=np.float64) - cur_xy
            dist = np.linalg.norm(direction)
            if dist > 1e-8:
                actions[i] = np.clip(direction / dist, -1.0, 1.0).astype(np.float32)

        return actions


world = swm.World(
    'swm/OGBMaze-v0',
    num_envs=NUM_ENVS,
    image_shape=IMAGE_SHAPE,
    loco_env_type='point',
    maze_env_type='maze',
    maze_type='large',
    ob_type='pixels',
    width=IMAGE_SHAPE[1],
    height=IMAGE_SHAPE[0],
    max_episode_steps=MAX_EPISODE_STEPS,
)

world.set_policy(PointMazeOraclePolicy())

world.collect(
    path=DATASET_PATH,
    episodes=NUM_EPISODES,
    seed=42,
)

print(f'Dataset saved to {DATASET_PATH}')
