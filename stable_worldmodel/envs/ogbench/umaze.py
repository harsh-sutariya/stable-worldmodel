"""U-maze layout for OGBench's point maze.

OGBench hardcodes its layouts in an if-chain inside ``make_maze_env`` with no
extension hook, so ``maze_type='umaze'`` is added by rewriting a *copy* of the
library source and executing it in a fresh namespace. The installed package is
never modified. If a future OGBench release changes the anchors the rewrite
relies on, :func:`build_make_maze_env` raises instead of silently misbehaving.
"""

import inspect

import ogbench.locomaze.maze as _ogb_maze

# 5x5 grid, 1 = wall. Two columns of free cells joined along the bottom row.
UMAZE_MAP = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

# (init_ij, goal_ij): the two arm tips, then arm-to-corner crossings.
UMAZE_TASKS = [
    [(1, 1), (1, 3)],
    [(1, 3), (1, 1)],
    [(1, 1), (3, 3)],
    [(1, 3), (3, 1)],
]

_ELSE = '            else:\n'
_RAISE = "                raise ValueError(f'Unknown maze type: {self._maze_type}')\n"
_SITES = 3  # MazeEnv map, MazeEnv tasks, BallEnv tasks (BallEnv is skipped)


def _branch(assignment: str) -> str:
    return f"            elif self._maze_type == 'umaze':\n{assignment}"


def _patch_source(src: str) -> str:
    sites = src.split(_ELSE + _RAISE)
    if len(sites) != _SITES + 1:
        raise RuntimeError(
            'Unsupported ogbench version: could not locate the maze-type '
            'dispatch sites needed to add the U-maze layout.'
        )

    rows = ''.join(f'                    {row},\n' for row in UMAZE_MAP)
    tasks = ''.join(f'                    {task},\n' for task in UMAZE_TASKS)
    branches = [
        _branch(f'                maze_map = [\n{rows}                ]\n'),
        _branch(f'                tasks = [\n{tasks}                ]\n'),
        '',
    ]
    out = sites[0]
    for branch, rest in zip(branches, sites[1:]):
        out += branch + _ELSE + _RAISE + rest
    return out


def build_make_maze_env():
    """Return ``make_maze_env`` with ``maze_type='umaze'`` available."""
    src = inspect.getsource(_ogb_maze)
    if "'umaze'" in src:  # already supported by the installed ogbench
        return _ogb_maze.make_maze_env
    namespace = dict(vars(_ogb_maze))
    exec(compile(_patch_source(src), _ogb_maze.__file__, 'exec'), namespace)
    return namespace['make_maze_env']
