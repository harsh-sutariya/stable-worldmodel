"""Tests for the in-repo U-maze layout added on top of OGBench."""

import inspect

import numpy as np
import pytest

ogb_maze = pytest.importorskip('ogbench.locomaze.maze')

from stable_worldmodel.envs.ogbench import umaze  # noqa: E402


def test_patch_adds_umaze_to_map_and_tasks_only():
    src = inspect.getsource(ogb_maze)
    if "'umaze'" in src:
        pytest.skip('installed ogbench already ships a U-maze')
    patched = umaze._patch_source(src)
    # MazeEnv map + MazeEnv tasks get the branch; BallEnv does not.
    assert patched.count("self._maze_type == 'umaze'") == 2


def test_patch_rejects_unrecognised_library_source():
    with pytest.raises(RuntimeError, match='Unsupported ogbench version'):
        umaze._patch_source('class Nothing: pass\n')


def test_umaze_env_layout_and_tasks():
    from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

    env = MazeEnv(maze_type='umaze', ob_type='pixels').env.unwrapped
    np.testing.assert_array_equal(env.maze_map, np.array(umaze.UMAZE_MAP))
    got = [(t['init_ij'], t['goal_ij']) for t in env.task_infos]
    assert got == [tuple(map(tuple, task)) for task in umaze.UMAZE_TASKS]


def test_unknown_maze_type_still_raises():
    from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

    with pytest.raises(ValueError, match='Unknown maze type'):
        MazeEnv(maze_type='nope', ob_type='pixels')
