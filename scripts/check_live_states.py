"""Check that live states (online RL) match states reconstructed from qpos/qvel (offline datasets).

Rolls out episodes in the given (possibly pixel-based) environment, replaying actions from the dataset, and compares
`compute_live_state` at every step with `reconstruct_state` applied to the qpos/qvel in `info`, computed in the
state-based version of the environment. The qpos/qvel are rounded to float32 first, as in the datasets.

Usage:
    python scripts/check_live_states.py --env_name=visual-cube-single-singletask-task1-v0
"""

import os
import sys

import gymnasium
import numpy as np
import ogbench
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.state_utils import GRIPPER_CONTACT_DIM, compute_live_state, reconstruct_state  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'visual-cube-single-singletask-task1-v0', 'Environment name.')
flags.DEFINE_string('dataset_name', 'cube-single-play-v0', 'Dataset to replay actions from.')
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes.')

CHECK_ATOL = 1e-5


def main(_):
    env = gymnasium.make(FLAGS.env_name)
    state_env = gymnasium.make(FLAGS.env_name.removeprefix('visual-'))
    state_env.reset(seed=0)

    file = np.load(os.path.join(os.path.expanduser(ogbench.utils.DEFAULT_DATASET_DIR), f'{FLAGS.dataset_name}.npz'))
    actions, terminals = file['actions'], file['terminals']
    episode_starts = np.concatenate([[0], np.nonzero(terminals)[0][:-1] + 1])

    live, reconstructed = [], []
    for episode in range(FLAGS.num_episodes):
        _, info = env.reset(seed=episode)
        t, is_initial, done = episode_starts[episode], True, False
        while True:
            live.append(compute_live_state(env))
            reconstructed.append(
                reconstruct_state(
                    state_env,
                    info['qpos'].astype(np.float32),
                    info['qvel'].astype(np.float32),
                    button_states=info.get('button_states'),
                    is_initial=is_initial,
                )
            )
            if done or terminals[t]:
                break
            _, _, terminated, truncated, info = env.step(actions[t])
            t, is_initial, done = t + 1, False, terminated or truncated

    live, reconstructed = np.array(live), np.array(reconstructed)
    err = np.abs(live - reconstructed)
    other_dims = [j for j in range(live.shape[1]) if j != GRIPPER_CONTACT_DIM]
    print(f'{len(live)} steps; contact in {live[:, GRIPPER_CONTACT_DIM].mean():.3f} of them')
    print(f'Max abs error (excluding gripper_contact): {err[:, other_dims].max():.3g} (tolerance {CHECK_ATOL})')
    agreement = np.mean(live[:, GRIPPER_CONTACT_DIM] == reconstructed[:, GRIPPER_CONTACT_DIM])
    print(f'gripper_contact agreement: {agreement:.5f}')
    passed = err[:, other_dims].max() <= CHECK_ATOL and agreement >= 0.999
    print('PASSED' if passed else 'FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    app.run(main)
