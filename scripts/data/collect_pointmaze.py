"""Collect PointMaze expert demonstrations using BFS oracle navigation.

Usage:
    uv run python scripts/data/collect_pointmaze.py

Note: Uses a manual collection loop (not world.collect) for two reasons:
1. Lance's RecordBatchReader callbacks run on a background thread which has no
   MuJoCo OpenGL context, causing a hang on the second episode.
2. EnvPool._write_env_info silently skips step-only keys (xy, qpos, qvel) that
   aren't in the initial reset info. We capture xy directly from the env.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.data.format import get_format
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
SEED = 42


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


def collect_to_memory(world, episodes, seed):
    """Run rollouts on the main thread and return a list of episode dicts.

    Captures xy directly from each env's unwrapped base_env.get_xy() because
    the World's stacked info dict only contains keys present at reset time.
    """
    buffers = [defaultdict(list) for _ in range(world.num_envs)]
    collected = []

    world.reset(seed=seed)
    next_seed = seed + world.num_envs

    def record_infos(infos, idxs):
        for col, data in infos.items():
            if col.startswith('_') or not isinstance(data, np.ndarray):
                continue
            if data.ndim > 1 and data.shape[1] == 1:
                data = np.squeeze(data, axis=1)
            for i in idxs:
                buffers[i][col].append(data[i].copy())

    def record_xy(idxs):
        for i in idxs:
            xy = np.array(world.envs.envs[i].unwrapped.get_xy(), dtype=np.float32)
            buffers[i]['xy'].append(xy)

    # Record initial reset observations
    record_infos(world.infos, range(world.num_envs))
    record_xy(range(world.num_envs))

    with tqdm(total=episodes, desc='Recording') as pbar:
        ep_count = 0

        while ep_count < episodes:
            actions = world.policy.get_action(world.infos)
            _, rews, terms, truncs, world.infos = world.envs.step(actions)
            world.rewards = rews
            world.terminateds = terms
            world.truncateds = truncs

            record_infos(world.infos, range(world.num_envs))
            record_xy(range(world.num_envs))

            done = terms | truncs
            if not done.any():
                continue

            for i in np.where(done)[0]:
                ep = {k: list(v) for k, v in buffers[i].items()}
                buffers[i].clear()
                if 'action' in ep:
                    ep['action'].append(ep['action'].pop(0))
                collected.append(ep)
                ep_count += 1
                pbar.update(1)
                if ep_count >= episodes:
                    break

            if ep_count >= episodes:
                break

            # Reset done envs with fresh seeds
            seeds = [None] * world.num_envs
            base = ep_count - int(done.sum())
            for rank, env_i in enumerate(np.where(done)[0]):
                seeds[env_i] = next_seed + base + rank
            _, world.infos = world.envs.reset(seed=seeds, mask=done)
            world.terminateds = terms.copy()
            world.truncateds = truncs.copy()
            world.terminateds[done] = False
            world.truncateds[done] = False
            reset_idxs = np.where(done)[0]
            record_infos(world.infos, reset_idxs)
            record_xy(reset_idxs)
            next_seed += int(done.sum())

    return collected


def main():
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

    print(f'Collecting {NUM_EPISODES} episodes with {NUM_ENVS} envs...', flush=True)
    episodes = collect_to_memory(world, NUM_EPISODES, seed=SEED)

    print(f'Collected {len(episodes)} episodes. Writing to {DATASET_PATH}...', flush=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_format('lance').open_writer(DATASET_PATH) as w:
        w.write_episodes(iter(episodes))
    print(f'Done. Dataset saved to {DATASET_PATH}', flush=True)


if __name__ == '__main__':
    main()
