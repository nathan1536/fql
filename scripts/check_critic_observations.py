"""Check the separate critic observations (memory-mapped frames from another camera) on a subset of a visual dataset.

With the same random state, a batch from the dataset with critic observations must equal
- in 'observations' (and all other keys): a batch from the dataset with the normal camera, and
- in 'critic_observations': the 'observations' of a batch from the dataset rendered with the critic's camera,
with frame stacking and augmentation (the same crops for the actor and the critic).

Usage (the subset is the one written by `scripts/check_dataset_equivalence.py`, re-rendered with
`scripts/render_dataset.py --dataset_dir=<subset_dir> --fovy_scale=0.35` and extracted with `scripts/npz_to_npy.py`):
    python scripts/check_critic_observations.py --subset_dir=<subset_dir>
"""

import os
import sys

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.camera_utils import camera_suffix  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.datasets import Dataset  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'visual-cube-single-play-singletask-task1-v0', 'Environment (dataset) name.')
flags.DEFINE_string('subset_dir', None, 'Directory of the subset datasets.')
flags.DEFINE_float('critic_fovy_scale', 0.35, 'Field-of-view scale of the critic camera.')
flags.DEFINE_integer('num_batches', 200, 'Number of batches to compare.')


def load(**kwargs):
    _, _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=3, dataset_dir=FLAGS.subset_dir, **kwargs
    )
    datasets = [Dataset.create(**train_dataset), val_dataset]
    for dataset in datasets:
        dataset.frame_stack, dataset.p_aug = 3, 0.5
    return datasets


def main(_):
    both = load(critic_obs_suffix=camera_suffix(FLAGS.critic_fovy_scale))
    actor_ref = load()
    critic_ref = load(fovy_scale=FLAGS.critic_fovy_scale)

    for split, (d, a_ref, c_ref) in enumerate(zip(both, actor_ref, critic_ref)):
        name = ['train', 'val'][split]
        np.random.seed(0)
        for i in range(FLAGS.num_batches):
            rng_state = np.random.get_state()
            batch = d.sample(256)
            np.random.set_state(rng_state)
            a = a_ref.sample(256)
            np.random.set_state(rng_state)
            c = c_ref.sample(256)
            assert set(batch) == set(a) | {'critic_observations', 'next_critic_observations'}, f'{name}: keys'
            for k in a:
                assert np.array_equal(batch[k], a[k]), f'{name} batch {i}: {k} differs from the normal camera'
            assert np.array_equal(batch['critic_observations'], c['observations']), f'{name} batch {i}: critic obs'
            assert np.array_equal(batch['next_critic_observations'], c['next_observations']), f'{name} batch {i}'
        print(f'{name}: {FLAGS.num_batches} batches: actor observations = normal camera, '
              f'critic observations = FOV x{FLAGS.critic_fovy_scale:g} camera (stacked, augmented)')
    print('PASSED')


if __name__ == '__main__':
    flags.mark_flag_as_required('subset_dir')
    app.run(main)
