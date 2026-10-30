"""Reconstruct privileged state vectors for an OGBench manipulation dataset (see `envs/state_utils.py`).

For each dataset file `<name>.npz` and `<name>-val.npz`, this writes `<name>-states.npz` and `<name>-val-states.npz`
next to it. Each contains 'states' with one row per row of the dataset file (aligned with its 'observations').

Usage:
    # Check against the stored observations of a state-based dataset.
    python scripts/reconstruct_states.py --dataset_name=cube-single-play-v0 --check
    # Reconstruct states for a pixel-based dataset.
    python scripts/reconstruct_states.py --dataset_name=visual-cube-single-play-v0
"""

import multiprocessing
import os
import sys
import time

import gymnasium
import numpy as np
import ogbench
from absl import app, flags
from ogbench.manipspace import lie

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.state_utils import GRIPPER_CONTACT_DIM, reconstruct_state  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'visual-cube-single-play-v0', 'OGBench dataset name (without the .npz suffix).')
flags.DEFINE_string('dataset_dir', ogbench.utils.DEFAULT_DATASET_DIR, 'Dataset directory.')
flags.DEFINE_integer('num_workers', 8, 'Number of worker processes.')
flags.DEFINE_integer('max_rows', None, 'Only process the first rows of each file (for quick tests).')
flags.DEFINE_bool('check', False, 'Compare with the stored observations (state-based datasets only).')
flags.DEFINE_bool('save', True, 'Whether to save the states.')

CHECK_ATOL = 1e-5  # Stored qpos/qvel are float32.
CHECK_MIN_CONTACT_AGREEMENT = 0.999
SINGULAR_YAW_RZZ = 1e-2

_env = None


def _init_worker(env_name):
    global _env
    _env = gymnasium.make(env_name)
    _env.reset(seed=0)


def _reconstruct_chunk(args):
    qpos, qvel, button_states, is_initial = args
    states = []
    for i in range(len(qpos)):
        states.append(
            reconstruct_state(
                _env,
                qpos[i],
                qvel[i],
                button_states=None if button_states is None else button_states[i],
                is_initial=is_initial[i],
            )
        )
    return np.array(states, dtype=np.float32)


def reconstruct_file(env_name, path, max_rows):
    file = np.load(path)
    qpos, qvel, terminals = file['qpos'], file['qvel'], file['terminals']
    button_states = file['button_states'] if 'button_states' in file else None
    if max_rows is not None:
        qpos, qvel, terminals = qpos[:max_rows], qvel[:max_rows], terminals[:max_rows]
        button_states = None if button_states is None else button_states[:max_rows]

    # In the stored files, `terminals` marks the last row of each trajectory, so the next row starts a new one.
    is_initial = np.concatenate([[True], terminals[:-1] > 0])

    chunks = np.array_split(np.arange(len(qpos)), FLAGS.num_workers * 16)
    chunk_args = [
        (qpos[c], qvel[c], None if button_states is None else button_states[c], is_initial[c]) for c in chunks
    ]
    with multiprocessing.get_context('spawn').Pool(
        FLAGS.num_workers, initializer=_init_worker, initargs=(env_name,)
    ) as pool:
        states = np.concatenate(pool.map(_reconstruct_chunk, chunk_args))
    return states, file


def singular_yaw_mask(env, qpos, shape):
    """Mask of the cube yaw dimensions (cos, sin) at rows where the cube's z-axis is nearly horizontal.

    The yaw is ill-defined there, so float32 rounding of the stored quaternion can change it arbitrarily.
    """
    mask = np.zeros(shape, dtype=bool)
    for i in range(getattr(env.unwrapped, '_num_cubes', 0)):
        quat_start = 14 + 7 * i + 3  # qpos: arm and gripper joints (14), then (xyz, wxyz) per cube.
        rzz = np.array([lie.SO3(wxyz=q).as_matrix()[2, 2] for q in qpos[:, quat_start : quat_start + 4]])
        yaw_dim = 19 + 9 * i + 7  # Observation: proprio (19), then (xyz, wxyz, cos, sin) per cube.
        mask[np.abs(rzz) < SINGULAR_YAW_RZZ, yaw_dim : yaw_dim + 2] = True
    return mask


def check(env, states, file, max_rows):
    obs = file['observations'][:max_rows] if max_rows is not None else file['observations']
    qpos = file['qpos'][:max_rows] if max_rows is not None else file['qpos']
    assert obs.shape == states.shape, f'Shape mismatch: stored {obs.shape} vs reconstructed {states.shape}'
    err = np.abs(states.astype(np.float64) - obs)
    mask = singular_yaw_mask(env, qpos.astype(np.float64), obs.shape)
    err[mask] = 0.0
    print(f'  Skipped cube yaw at {mask.any(axis=1).sum()} rows with an ill-defined yaw (|R_zz| < {SINGULAR_YAW_RZZ})')

    other_dims = [j for j in range(obs.shape[1]) if j != GRIPPER_CONTACT_DIM]
    max_err = err[:, other_dims].max(axis=0)
    print('  Max abs error per dimension (excluding gripper_contact):')
    print('  ' + np.array2string(max_err, precision=2, max_line_width=110).replace('\n', '\n  '))
    worst = other_dims[int(np.argmax(max_err))]
    print(f'  Worst dimension: {worst} (max abs error {max_err.max():.3g}, tolerance {CHECK_ATOL})')

    agreement = np.mean((obs[:, GRIPPER_CONTACT_DIM] > 0) == (states[:, GRIPPER_CONTACT_DIM] > 0))
    print(f'  gripper_contact > 0 agreement: {agreement:.5f} (minimum {CHECK_MIN_CONTACT_AGREEMENT})')

    return bool(max_err.max() <= CHECK_ATOL and agreement >= CHECK_MIN_CONTACT_AGREEMENT)


def main(_):
    splits = FLAGS.dataset_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
    # Pixel-based environments only differ in rendering, so use the state-based one (no rendering needed).
    env_name = env_name.removeprefix('visual-')
    dataset_dir = os.path.expanduser(FLAGS.dataset_dir)
    assert FLAGS.max_rows is None or not FLAGS.save, 'Use --nosave with --max_rows.'
    ogbench.download_datasets([FLAGS.dataset_name], dataset_dir)

    env = gymnasium.make(env_name)
    passed = True
    for suffix in ['', '-val']:
        path = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{suffix}.npz')
        start_time = time.time()
        states, file = reconstruct_file(env_name, path, FLAGS.max_rows)
        print(f'{os.path.basename(path)}: {len(states)} states of dimension {states.shape[1]} '
              f'in {time.time() - start_time:.0f}s', flush=True)

        if FLAGS.check:
            passed &= check(env, states, file, FLAGS.max_rows)
        if FLAGS.save:
            out_path = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{suffix}-states.npz')
            np.savez(out_path, states=states)
            print(f'  Saved: {out_path}')

    if FLAGS.check:
        print('PASSED' if passed else 'FAILED')
        sys.exit(0 if passed else 1)


if __name__ == '__main__':
    app.run(main)
