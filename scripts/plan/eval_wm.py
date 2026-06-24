"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


def eval_model(cfg: DictConfig, model=None) -> dict:
    """Run CEM planning evaluation and return metrics dict.

    If model is provided it is used directly (skips load_pretrained).
    cfg must follow the same structure as the Hydra tworoom/pusht configs.
    Returns {'success_rate': float, 'episode_successes': array, ...}
    """
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal':   img_transform(cfg, img_dtype),
    }

    dataset     = get_dataset(cfg, cfg.eval.dataset_name)
    col_name    = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data  = dataset.get_col_data(col)
        col_data  = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    policy_name = cfg.get('policy', 'random')
    device      = cfg.get('device', 'cpu')

    if policy_name != 'random':
        if model is None:
            model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = 'backbone' if hasattr(model, 'backbone') else 'encoder'
            setattr(model, encoder_attr, torch.compile(getattr(model, encoder_attr)))
            model.predictor = torch.compile(model.predictor)
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver      = hydra.utils.instantiate(cfg.solver, model=model)
        policy      = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy).parent
        if policy_name != 'random'
        else Path(__file__).parent
    )

    episode_len    = get_episodes_length(dataset, ep_indices)
    max_start_idx  = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask    = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = np.sort(
        valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)]
    )

    eval_episodes  = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError('Not enough episodes with sufficient length for evaluation.')

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)
    print(f'[eval] saving videos to {results_path.resolve()}')

    use_bf16       = cfg.get('bf16', False)
    autocast_device = device if device == 'cuda' else 'cpu'

    callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)

    start_time = time.time()
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_bf16):
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            video=results_path,
        )
    end_time = time.time()

    metrics['evaluation_time'] = end_time - start_time
    print(metrics)
    return metrics


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    metrics = eval_model(cfg)

    results_path = (
        Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy).parent
        if cfg.get('policy', 'random') != 'random'
        else Path(__file__).parent
    )
    out_file = results_path / cfg.output.filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open('a') as f:
        f.write('\n')
        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {metrics.get("evaluation_time")} seconds\n')

    print(f'[eval] results saved to {out_file}')


if __name__ == '__main__':
    run()
