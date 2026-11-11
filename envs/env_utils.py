import collections
import os
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box
from ogbench.relabel_utils import relabel_dataset

from envs.camera_utils import CameraWrapper, camera_suffix
from utils.datasets import Dataset, IndexedArray


def load_shared_obs_dataset(dataset_path):
    """Load an OGBench dataset without duplicating observations.

    Equivalent to `ogbench.utils.load_dataset(dataset_path, ob_dtype=np.uint8, add_info=True)`, except that
    'observations' and 'next_observations' are `IndexedArray` views into a single copy of the stored observations.
    The regular loader keeps three copies in memory at its peak, which does not fit for visual datasets.
    """
    file = np.load(dataset_path)

    obs = file['observations'][...].astype(np.uint8, copy=False)
    actions = file['actions'][...].astype(np.float32, copy=False)
    terminals = file['terminals'][...].astype(np.float32, copy=False)

    ob_idxs, next_ob_idxs = get_transition_idxs(terminals)
    new_terminals = np.concatenate([terminals[1:], [1.0]])

    dataset = dict(
        observations=IndexedArray(obs, ob_idxs),
        next_observations=IndexedArray(obs, next_ob_idxs),
        actions=actions[ob_idxs],
        terminals=new_terminals[ob_idxs].astype(np.float32),
    )
    for k in ['qpos', 'qvel', 'button_states']:
        if k in file:
            dataset[k] = file[k][...][ob_idxs]

    return dataset


def get_transition_idxs(terminals):
    """Return the rows of the observations and next observations of each transition in an OGBench dataset file.

    These are the same masks as in the regular (non-compact) `ogbench.utils.load_dataset`.
    """
    ob_mask = (1.0 - terminals).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    return np.nonzero(ob_mask)[0], np.nonzero(next_ob_mask)[0]


def add_states_to_dataset(dataset, dataset_path):
    """Add 'states' and 'next_states' from `<dataset>-states.npz` (see `scripts/reconstruct_states.py`)."""
    states_path = dataset_path.removesuffix('.npz') + '-states.npz'
    if not os.path.exists(states_path):
        dataset_name = os.path.basename(dataset_path).removesuffix('.npz').removesuffix('-val')
        raise FileNotFoundError(
            f'{states_path} not found. Run: python scripts/reconstruct_states.py --dataset_name={dataset_name}'
        )
    states = np.load(states_path)['states']
    terminals = np.load(dataset_path)['terminals']
    assert len(states) == len(terminals), f'{states_path} does not match {dataset_path}.'

    ob_idxs, next_ob_idxs = get_transition_idxs(terminals)
    dataset['states'] = IndexedArray(states, ob_idxs)
    dataset['next_states'] = IndexedArray(states, next_ob_idxs)


def add_critic_observations_to_dataset(dataset, dataset_path, critic_obs_path):
    """Add 'critic_observations' and 'next_critic_observations' for a separate critic input (e.g., another camera).

    `critic_obs_path` is an .npy file with one frame per row of the dataset file (see `scripts/npz_to_npy.py`). It is
    memory-mapped, so its frames do not need to fit in memory.
    """
    if not os.path.exists(critic_obs_path):
        raise FileNotFoundError(f'{critic_obs_path} not found. Run scripts/render_dataset.py, scripts/npz_to_npy.py.')
    frames = np.load(critic_obs_path, mmap_mode='r')
    terminals = np.load(dataset_path)['terminals']
    assert len(frames) == len(terminals), f'{critic_obs_path} does not match {dataset_path}.'

    ob_idxs, next_ob_idxs = get_transition_idxs(terminals)
    dataset['critic_observations'] = IndexedArray(frames, ob_idxs)
    dataset['next_critic_observations'] = IndexedArray(frames, next_ob_idxs)


def get_ogbench_names(dataset_name):
    """Return the environment and dataset file names of an OGBench single-task dataset name."""
    splits = dataset_name.split('-')
    pos = splits.index('singletask')
    env_name = '-'.join(splits[: pos - 1] + splits[pos:])  # Remove the dataset type.
    dataset_name = '-'.join(splits[:pos] + splits[-1:])  # Remove the words 'singletask' and 'task\d' (if exists).
    return env_name, dataset_name


def make_ogbench_visual_env_and_datasets(dataset_name, dataset_dir=ogbench.utils.DEFAULT_DATASET_DIR, fovy_scale=None):
    """Memory-efficient version of `ogbench.make_env_and_datasets` for visual single-task datasets.

    With `fovy_scale`, the datasets are the ones re-rendered with that camera (`scripts/render_dataset.py`).
    """
    env_name, dataset_name = get_ogbench_names(dataset_name)
    env = gymnasium.make(env_name)

    dataset_dir = os.path.expanduser(dataset_dir)
    suffix = camera_suffix(fovy_scale)
    if fovy_scale is None:
        ogbench.download_datasets([dataset_name], dataset_dir)
    elif not os.path.exists(os.path.join(dataset_dir, f'{dataset_name}{suffix}.npz')):
        raise FileNotFoundError(
            f'Run: python scripts/render_dataset.py --dataset_name={dataset_name} --fovy_scale={fovy_scale:g}'
        )
    train_dataset = load_shared_obs_dataset(os.path.join(dataset_dir, f'{dataset_name}{suffix}.npz'))
    val_dataset = load_shared_obs_dataset(os.path.join(dataset_dir, f'{dataset_name}{suffix}-val.npz'))

    # Add reward information, then remove the information keys (as with `add_info=False`).
    for dataset in [train_dataset, val_dataset]:
        relabel_dataset(env_name, env, dataset)
        for k in ['qpos', 'qvel', 'button_states']:
            dataset.pop(k, None)

    return env, train_dataset, val_dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(
    env_name,
    frame_stack=None,
    action_clip_eps=1e-5,
    dataset_dir=ogbench.utils.DEFAULT_DATASET_DIR,
    add_states=False,
    fovy_scale=None,
    critic_obs_suffix=None,
):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        dataset_dir: Directory of the OGBench datasets.
        add_states: Whether to add privileged 'states' and 'next_states' to the datasets (OGBench manipulation only).
        fovy_scale: Scale of the field of view of the pixel camera, for the environments and (re-rendered) datasets
            (OGBench pixel-based manipulation only).
        critic_obs_suffix: If given, add 'critic_observations' from the re-rendered dataset with this suffix (e.g.,
            '-fovy0.35' or '-nolockcolor'; see `scripts/render_dataset.py`) to the datasets (OGBench pixel-based
            manipulation only). The environments are unchanged, since only the actor acts.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """
    if add_states:
        assert 'singletask' in env_name, 'States are only supported for OGBench single-task datasets.'
    if fovy_scale is not None or critic_obs_suffix is not None:
        assert 'singletask' in env_name and 'visual' in env_name, 'Cameras only change for OGBench pixel datasets.'

    if 'singletask' in env_name:
        # OGBench.
        if 'visual' in env_name:
            env, train_dataset, val_dataset = make_ogbench_visual_env_and_datasets(
                env_name, dataset_dir=dataset_dir, fovy_scale=fovy_scale
            )
        else:
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, dataset_dir=dataset_dir)
        if add_states:
            dataset_path = os.path.join(
                os.path.expanduser(dataset_dir), get_ogbench_names(env_name)[1] + camera_suffix(fovy_scale)
            )
            add_states_to_dataset(train_dataset, f'{dataset_path}.npz')
            add_states_to_dataset(val_dataset, f'{dataset_path}-val.npz')
        if critic_obs_suffix is not None:
            data_dir, dataset_name = os.path.expanduser(dataset_dir), get_ogbench_names(env_name)[1]
            dataset_path = os.path.join(data_dir, dataset_name + camera_suffix(fovy_scale))
            critic_path = os.path.join(data_dir, dataset_name + critic_obs_suffix)
            add_critic_observations_to_dataset(train_dataset, f'{dataset_path}.npz', f'{critic_path}-observations.npy')
            add_critic_observations_to_dataset(
                val_dataset, f'{dataset_path}-val.npz', f'{critic_path}-val-observations.npy'
            )
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if fovy_scale is not None:
            env, eval_env = CameraWrapper(env, fovy_scale), CameraWrapper(eval_env, fovy_scale)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
