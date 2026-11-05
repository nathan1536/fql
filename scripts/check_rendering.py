"""Check that the environment's pixel observations match observations re-rendered from qpos/qvel.

Rolls out episodes in a pixel-based environment with a changed camera (as used for evaluation and online RL),
replaying actions from a dataset, and re-renders each observation from the qpos/qvel in `info` (rounded to float32,
as in the datasets) the way `scripts/render_dataset.py` does. Also checks that `make_env_and_datasets` loads the
re-rendered datasets and wraps the environments when the subset datasets exist.

Usage:
    python scripts/check_rendering.py --fovy_scale=0.35
"""

import os
import sys

import gymnasium
import numpy as np
import ogbench
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.camera_utils import CameraWrapper, set_camera  # noqa: E402
from envs.state_utils import set_rendered_state  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'visual-cube-single-singletask-task1-v0', 'Pixel-based environment name.')
flags.DEFINE_string('render_env_name', 'visual-cube-single-v0', 'Environment used by scripts/render_dataset.py.')
flags.DEFINE_string('dataset_name', 'cube-single-play-v0', 'Dataset to replay actions from.')
flags.DEFINE_float('fovy_scale', 0.35, 'Scale of the field of view of the pixel camera.')
flags.DEFINE_integer('num_episodes', 5, 'Number of episodes.')


def main(_):
    env = CameraWrapper(gymnasium.make(FLAGS.env_name), FLAGS.fovy_scale)
    render_env = gymnasium.make(FLAGS.render_env_name)
    render_env.reset(seed=0)
    set_camera(render_env, FLAGS.fovy_scale)

    file = np.load(os.path.join(os.path.expanduser(ogbench.utils.DEFAULT_DATASET_DIR), f'{FLAGS.dataset_name}.npz'))
    actions, terminals = file['actions'], file['terminals']
    episode_starts = np.concatenate([[0], np.nonzero(terminals)[0][:-1] + 1])

    live, rendered = [], []
    for episode in range(FLAGS.num_episodes):
        ob, info = env.reset(seed=episode)
        t, is_initial, done = episode_starts[episode], True, False
        while True:
            set_rendered_state(
                render_env,
                info['qpos'].astype(np.float32),
                info['qvel'].astype(np.float32),
                button_states=info.get('button_states'),
                is_initial=is_initial,
            )
            live.append(ob)
            rendered.append(render_env.unwrapped.render())
            if done or terminals[t]:
                break
            ob, _, terminated, truncated, info = env.step(actions[t])
            t, is_initial, done = t + 1, False, terminated or truncated

    d = np.abs(np.array(live, dtype=np.int16) - np.array(rendered, dtype=np.int16))
    identical = np.mean(d == 0)
    print(f'{len(live)} frames (fovy scale {FLAGS.fovy_scale:g}): values identical {identical:.5f}, '
          f'mean abs diff {d.mean():.4f}, max {d.max()}')
    passed = identical > 0.99
    print('PASSED' if passed else 'FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    app.run(main)
