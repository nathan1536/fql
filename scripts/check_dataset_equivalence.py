"""Check that the memory-efficient visual dataset pipeline matches the original one exactly.

The original pipeline (`ogbench.make_env_and_datasets` + copying into a `ReplayBuffer`) does not fit in memory for full
visual datasets, so this script compares both pipelines on a subset of the first `--num_trajs` trajectories.

Usage:
    python scripts/check_dataset_equivalence.py --env_name=visual-cube-single-play-singletask-task1-v0
"""

import os
import sys

import numpy as np
import ogbench
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.datasets import Dataset, ReplayBuffer  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'visual-cube-single-play-singletask-task1-v0', 'Environment (dataset) name.')
flags.DEFINE_string('subset_dir', None, 'Directory to write the subset datasets to.')
flags.DEFINE_integer('num_trajs', 20, 'Number of trajectories in the subset.')
flags.DEFINE_integer('num_batches', 1000, 'Number of batches to compare.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('frame_stack', 3, 'Number of frames to stack.')
flags.DEFINE_float('p_aug', 0.5, 'Probability of applying image augmentation.')


def write_subset(src_path, dst_path, num_trajs):
    """Write the first `num_trajs` trajectories of an OGBench dataset file."""
    file = np.load(src_path)
    end = np.nonzero(file['terminals'] > 0)[0][num_trajs - 1] + 1
    np.savez(dst_path, **{k: file[k][:end] for k in file.files})


def original_pipeline(env_name, dataset_dir):
    """The pipeline of the original `envs/env_utils.py` and `main.py` (offline RL, no balanced sampling)."""
    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, dataset_dir=dataset_dir)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    action_clip_eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
    )
    val_dataset = val_dataset.copy(
        add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
    )
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset), size=train_dataset.size + 1)
    return train_dataset, val_dataset


def new_pipeline(env_name, dataset_dir):
    """The pipeline of the current `envs/env_utils.py` and `main.py` (offline RL, no balanced sampling)."""
    _, _, train_dataset, val_dataset = make_env_and_datasets(env_name, dataset_dir=dataset_dir)
    train_dataset = Dataset.create(**train_dataset)
    return train_dataset, val_dataset


def compare_dicts(name, a, b):
    assert set(a.keys()) == set(b.keys()), f'{name}: keys differ: {sorted(a.keys())} vs {sorted(b.keys())}'
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype, f'{name}/{k}: dtype {x.dtype} vs {y.dtype}'
        assert x.shape == y.shape, f'{name}/{k}: shape {x.shape} vs {y.shape}'
        assert np.array_equal(x, y), f'{name}/{k}: values differ'


def main(_):
    splits = FLAGS.env_name.split('-')
    pos = splits.index('singletask')
    dataset_name = '-'.join(splits[:pos] + splits[-1:])

    ogbench.download_datasets([dataset_name])
    src_dir = os.path.expanduser(ogbench.utils.DEFAULT_DATASET_DIR)
    os.makedirs(FLAGS.subset_dir, exist_ok=True)
    for suffix in ['', '-val']:
        dst_path = os.path.join(FLAGS.subset_dir, f'{dataset_name}{suffix}.npz')
        if not os.path.exists(dst_path):
            print(f'Writing subset: {dst_path}', flush=True)
            write_subset(os.path.join(src_dir, f'{dataset_name}{suffix}.npz'), dst_path, FLAGS.num_trajs)

    old_train, old_val = original_pipeline(FLAGS.env_name, FLAGS.subset_dir)
    new_train, new_val = new_pipeline(FLAGS.env_name, FLAGS.subset_dir)
    print(f'Train size: {old_train.size} (original) vs {new_train.size} (new)')
    assert old_train.size == new_train.size and old_val.size == new_val.size
    assert np.array_equal(old_train.terminal_locs, new_train.terminal_locs)
    assert np.array_equal(old_train.initial_locs, new_train.initial_locs)

    # Full datasets.
    compare_dicts('train', {k: v[: old_train.size] for k, v in old_train.items()}, dict(new_train))
    compare_dicts('val', dict(old_val), dict(new_val))
    print('Full datasets: identical')

    # Sampled batches with frame stacking and augmentation, using the same random seed.
    for dataset in [old_train, old_val, new_train, new_val]:
        dataset.frame_stack = FLAGS.frame_stack
        dataset.p_aug = FLAGS.p_aug
    for old, new, name in [(old_train, new_train, 'train'), (old_val, new_val, 'val')]:
        np.random.seed(0)
        for i in range(FLAGS.num_batches):
            # Draw both batches from the same random state, one batch at a time to bound memory.
            rng_state = np.random.get_state()
            a = old.sample(FLAGS.batch_size)
            np.random.set_state(rng_state)
            b = new.sample(FLAGS.batch_size)
            compare_dicts(f'{name}_batch_{i}', a, b)
        print(f'{FLAGS.num_batches} sampled {name} batches: identical')

    print('PASSED')


if __name__ == '__main__':
    flags.mark_flag_as_required('subset_dir')
    app.run(main)
