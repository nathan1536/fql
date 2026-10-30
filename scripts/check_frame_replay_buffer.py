"""Check `FrameReplayBuffer` and the online loop's transitions on a small subset of a visual dataset.

Rolls out episodes in the environment the same way as the online loop in `main.py` (random actions), then checks:
1. Sampling from the buffer gives exactly the same batches as a plain dataset with all frames written out.
2. The stacked observations sampled for each online transition equal the stacked observations the agent saw.
3. The stored states match states reconstructed from the qpos/qvel in `info`, and `next_states[t] == states[t + 1]`.

Usage (the subset is the one written by `scripts/check_dataset_equivalence.py`):
    python scripts/reconstruct_states.py --dataset_name=visual-cube-single-play-v0 --dataset_dir=<subset_dir>
    python scripts/check_frame_replay_buffer.py --subset_dir=<subset_dir>
"""

import os
import shutil
import sys
import tempfile

import gymnasium
import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_utils import get_ogbench_names, make_env_and_datasets  # noqa: E402
from envs.state_utils import compute_live_state, reconstruct_state  # noqa: E402
from utils.datasets import Dataset, FrameReplayBuffer  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'visual-cube-single-play-singletask-task1-v0', 'Environment (dataset) name.')
flags.DEFINE_string('subset_dir', None, 'Directory of the subset datasets (with states).')
flags.DEFINE_integer('num_episodes', 3, 'Number of online episodes.')
flags.DEFINE_integer('frame_stack', 3, 'Number of frames to stack.')
flags.DEFINE_integer('num_batches', 200, 'Number of batches to compare.')


def collect(env, num_episodes, frame_channels):
    """Collect transitions like the online loop in `main.py`, plus what the agent saw and the info qpos/qvel.

    Whole episodes are collected because `Dataset` (the reference) assumes that the data ends with a terminal.
    """
    transitions, seen, infos = [], [], []
    done = True
    while not (done and len(transitions) > 0 and sum(t['terminals'] for t in transitions) == num_episodes):
        if done:
            ob, info = env.reset()
            state = compute_live_state(env)
            is_initial = True
        action = env.action_space.sample().astype(np.float32)
        next_ob, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_state = compute_live_state(env)
        transitions.append(
            dict(
                observations=ob[..., -frame_channels:],
                actions=action,
                rewards=reward,
                terminals=float(done),
                masks=1.0 - terminated,
                next_observations=next_ob[..., -frame_channels:],
                states=state,
                next_states=next_state,
            )
        )
        seen.append((ob, next_ob))
        infos.append((info['qpos'], info['qvel'], is_initial))
        ob, state, info, is_initial = next_ob, next_state, next_info, False
    return transitions, seen, infos


def compare(name, a, b):
    assert set(a) == set(b), f'{name}: keys differ'
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f'{name}/{k}: values differ'


def main(_):
    env, _, train_dataset, _ = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, dataset_dir=FLAGS.subset_dir, add_states=True
    )
    train_dataset = Dataset.create(**train_dataset)
    frame_channels = train_dataset['observations'].shape[-1]
    n = train_dataset.size

    env.action_space.seed(0)
    transitions, seen, infos = collect(env, FLAGS.num_episodes, frame_channels)
    num_online = len(transitions)
    print(f'Collected {num_online} online transitions ({FLAGS.num_episodes} episodes)')

    # Buffer as in `main.py` (dataset + online transitions), and a balanced-sampling buffer (online only).
    buffer = FrameReplayBuffer.create(num_online, init_dataset=train_dataset)
    example_transition = {k: v[0] for k, v in train_dataset.items()}
    online_buffer = FrameReplayBuffer.create(num_online, example_transition=example_transition)
    # The same buffers with new frames stored in a file.
    frames_dir = tempfile.mkdtemp()
    disk_buffer = FrameReplayBuffer.create(
        num_online, init_dataset=train_dataset, frames_path=os.path.join(frames_dir, 'frames.dat')
    )
    disk_online_buffer = FrameReplayBuffer.create(
        num_online, example_transition=example_transition, frames_path=os.path.join(frames_dir, 'online_frames.dat')
    )
    for t in transitions:
        for b in [buffer, online_buffer, disk_buffer, disk_online_buffer]:
            b.add_transition(t)

    # Reference: a plain dataset with every frame written out.
    reference = {k: np.asarray(v) for k, v in train_dataset.items()}
    for k in reference:
        reference[k] = np.concatenate([reference[k], np.array([t[k] for t in transitions], dtype=reference[k].dtype)])
    reference = Dataset.create(**reference)
    assert reference.size == buffer.size == n + num_online
    assert np.array_equal(reference.initial_locs, buffer.initial_locs), 'Episode starts differ.'

    # 1. Same batches as the reference, with frame stacking and augmentation.
    for dataset in [buffer, disk_buffer, reference]:
        dataset.frame_stack = FLAGS.frame_stack
        dataset.p_aug = 0.5
    np.random.seed(0)
    for i in range(FLAGS.num_batches):
        rng_state = np.random.get_state()
        b = reference.sample(256)
        for name, dataset in [('memory', buffer), ('disk', disk_buffer)]:
            np.random.set_state(rng_state)
            compare(f'{name}_batch_{i}', dataset.sample(256), b)
    print(f'1. {FLAGS.num_batches} sampled batches (frames in memory and on disk): identical to the reference dataset')

    # 2. Stacked observations of the online transitions equal what the agent saw.
    for dataset, offset, name in [
        (buffer, n, 'dataset + online'),
        (online_buffer, 0, 'online only'),
        (disk_buffer, n, 'dataset + online, disk'),
        (disk_online_buffer, 0, 'online only, disk'),
    ]:
        dataset.frame_stack = FLAGS.frame_stack
        dataset.p_aug = None
        batch = dataset.sample(num_online, idxs=np.arange(offset, offset + num_online))
        assert np.array_equal(batch['observations'], np.array([s[0] for s in seen])), f'{name}: observations'
        assert np.array_equal(batch['next_observations'], np.array([s[1] for s in seen])), f'{name}: next obs'
    print('2. Stacked observations of all online transitions: identical to what the agent saw')

    # 3. States match the info qpos/qvel, and next states match the next transition's states.
    state_env = gymnasium.make(get_ogbench_names(FLAGS.env_name)[0].removeprefix('visual-'))
    state_env.reset(seed=0)
    states = np.asarray(buffer['states'][n : n + num_online])
    next_states = np.asarray(buffer['next_states'][n : n + num_online])
    reconstructed = np.array(
        [reconstruct_state(state_env, q.astype(np.float32), v.astype(np.float32), is_initial=i) for q, v, i in infos]
    )
    max_err = np.abs(states - reconstructed).max()
    assert max_err < 1e-5, f'States differ from reconstruction (max abs error {max_err})'
    not_terminal = np.array([t['terminals'] == 0 for t in transitions])[:-1]
    assert np.array_equal(next_states[:-1][not_terminal], states[1:][not_terminal]), 'next_states[t] != states[t + 1]'
    print(f'3. States: match reconstruction from info (max abs error {max_err:.2g}); next_states consistent')

    shutil.rmtree(frames_dir)
    print('PASSED')


if __name__ == '__main__':
    flags.mark_flag_as_required('subset_dir')
    app.run(main)
