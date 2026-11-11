"""Re-render the pixel observations of an OGBench manipulation dataset with a changed camera or colors.

The stored qpos/qvel (and button states) are written into the simulator and the pixel camera is rendered, as in
`scripts/reconstruct_states.py`. For each dataset file `<name>.npz` and `<name>-val.npz`, this writes
`<name><suffix>.npz` and `<name><suffix>-val.npz` (suffix e.g. '-fovy0.35' or '-nolockcolor') with the new
observations and all other keys unchanged, links the state files of the original dataset (the physics is unchanged),
and saves how often the cube and the gripper are not visible.

Usage:
    # Check the re-rendering against the official images (small subsets only: loads the official images).
    python scripts/render_dataset.py --dataset_dir=<subset_dir> --fovy_scale=1 --check
    # Render a dataset with a narrower field of view (zoomed in).
    python scripts/render_dataset.py --dataset_name=visual-cube-single-play-v0 --fovy_scale=0.35
    # Render a scene dataset without the lock colors of the buttons and handles.
    python scripts/render_dataset.py --dataset_name=visual-scene-play-v0 --hide_lock_colors
"""

import json
import multiprocessing
import os
import sys
import time
import zipfile

import gymnasium
import mujoco
import numpy as np
import ogbench
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.camera_utils import camera_suffix, hide_lock_colors, set_camera  # noqa: E402
from envs.state_utils import set_rendered_state  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'visual-cube-single-play-v0', 'OGBench pixel-based dataset name.')
flags.DEFINE_string('dataset_dir', ogbench.utils.DEFAULT_DATASET_DIR, 'Dataset directory.')
flags.DEFINE_float('fovy_scale', None, 'Scale of the field of view of the pixel camera (unchanged if None).')
flags.DEFINE_bool('hide_lock_colors', False, 'Render scene buttons and handles without their lock colors.')
flags.DEFINE_integer('num_workers', 6, 'Number of worker processes.')
flags.DEFINE_integer('chunk_size', 2000, 'Number of frames rendered per task.')
flags.DEFINE_integer('visibility_every', 10, 'Measure visibility on every n-th frame.')
flags.DEFINE_bool('check', False, 'Compare with the stored observations (loads them; use small datasets only).')

_env = None
_visibility_every = None
_hide_lock_colors = False


def _init_worker(env_name, fovy_scale, hide_lock, visibility_every):
    global _env, _visibility_every, _hide_lock_colors
    _visibility_every = visibility_every
    _hide_lock_colors = hide_lock
    _env = gymnasium.make(env_name)
    _env.reset(seed=0)
    if fovy_scale is not None:
        set_camera(_env, fovy_scale)


def _geom_ids(u):
    """Geom ids of the cubes and of the gripper."""
    m = u._model
    cubes = {g for ids in getattr(u, '_cube_geom_ids_list', []) for g in ids}
    gripper_bodies = {b for b in range(m.nbody) if 'robotiq' in m.body(b).name}
    gripper = {g for g in range(m.ngeom) if m.geom_bodyid[g] in gripper_bodies}
    return np.array(sorted(cubes), dtype=np.int64), np.array(sorted(gripper), dtype=np.int64)


def _render_chunk(args):
    start, qpos, qvel, button_states, is_initial = args
    u = _env.unwrapped
    cubes, gripper = _geom_ids(u)
    frames, visibility = [], []
    for i in range(len(qpos)):
        set_rendered_state(
            _env, qpos[i], qvel[i], None if button_states is None else button_states[i], is_initial=is_initial[i]
        )
        if _hide_lock_colors:
            hide_lock_colors(_env)  # Setting the state sets the lock colors again.
        frames.append(u.render())
        if (start + i) % _visibility_every == 0:
            seg = u.render(segmentation=True)
            geoms = seg[..., 0][seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM]
            visibility.append((np.isin(cubes, geoms).any(), np.isin(gripper, geoms).any()))
    return np.array(frames, dtype=np.uint8), visibility


def _write_array(zf, name, arr):
    with zf.open(f'{name}.npy', 'w', force_zip64=True) as f:
        np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


def render_file(env_name, src_path, dst_path):
    file = np.load(src_path)
    qpos, qvel, terminals = file['qpos'], file['qvel'], file['terminals']
    button_states = file['button_states'] if 'button_states' in file else None
    is_initial = np.concatenate([[True], terminals[:-1] > 0])  # `terminals` marks the last row of each trajectory.
    n = len(qpos)

    with zipfile.ZipFile(dst_path + '.tmp', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for k in file.files:
            if k != 'observations':
                _write_array(zf, k, file[k])

        chunks = [np.arange(s, min(s + FLAGS.chunk_size, n)) for s in range(0, n, FLAGS.chunk_size)]
        tasks = [
            (c[0], qpos[c], qvel[c], None if button_states is None else button_states[c], is_initial[c]) for c in chunks
        ]
        visibility, num_written = [], 0
        with zf.open('observations.npy', 'w', force_zip64=True) as f:
            header = {'descr': np.lib.format.dtype_to_descr(np.dtype(np.uint8)), 'fortran_order': False}
            header['shape'] = None  # Set below, once the frame shape is known.
            with multiprocessing.get_context('spawn').Pool(
                FLAGS.num_workers,
                initializer=_init_worker,
                initargs=(env_name, FLAGS.fovy_scale, FLAGS.hide_lock_colors, FLAGS.visibility_every),
            ) as pool:
                # Render a bounded number of chunks at a time to bound memory.
                for b in range(0, len(tasks), 4 * FLAGS.num_workers):
                    for frames, vis in pool.map(_render_chunk, tasks[b : b + 4 * FLAGS.num_workers]):
                        if header['shape'] is None:
                            header['shape'] = (n, *frames.shape[1:])
                            np.lib.format.write_array_header_1_0(f, header)
                        f.write(frames.tobytes())
                        num_written += len(frames)
                        visibility.extend(vis)
                    print(f'  {num_written}/{n} frames', flush=True)
    os.replace(dst_path + '.tmp', dst_path)

    visibility = np.array(visibility)
    return dict(cube_hidden=float(1 - visibility[:, 0].mean()), gripper_hidden=float(1 - visibility[:, 1].mean()))


def check(src_path, dst_path):
    """Compare the re-rendered observations with the stored ones (excluding the top row of pixels)."""
    stored, rendered = np.load(src_path)['observations'], np.load(dst_path)['observations']
    d = np.abs(stored.astype(np.int16) - rendered.astype(np.int16))[:, 1:]
    print(f'  Check (excluding the top row of pixels): values identical {np.mean(d == 0):.5f}, '
          f'mean abs diff {d.mean():.4f}, max {d.max()}')
    return float(np.mean(d == 0))


def main(_):
    splits = FLAGS.dataset_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
    suffix = camera_suffix(FLAGS.fovy_scale) + ('-nolockcolor' if FLAGS.hide_lock_colors else '')
    assert suffix, 'Change the camera or the colors (the output must not overwrite the original dataset).'
    dataset_dir = os.path.expanduser(FLAGS.dataset_dir)

    stats = {}
    for split in ['', '-val']:
        src_path = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{split}.npz')
        dst_path = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{suffix}{split}.npz')
        start_time = time.time()
        print(f'Rendering {os.path.basename(src_path)} -> {os.path.basename(dst_path)}', flush=True)
        stats[split.lstrip('-') or 'train'] = render_file(env_name, src_path, dst_path)
        print(f'  Done in {time.time() - start_time:.0f}s; {stats[split.lstrip("-") or "train"]}', flush=True)
        if FLAGS.check:
            stats[split.lstrip('-') or 'train']['values_identical'] = check(src_path, dst_path)

        # The physics is unchanged, so the states are those of the original dataset.
        src_states = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{split}-states.npz')
        dst_states = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{suffix}{split}-states.npz')
        if os.path.exists(src_states) and not os.path.exists(dst_states):
            os.symlink(os.path.basename(src_states), dst_states)

    stats_path = os.path.join(dataset_dir, f'{FLAGS.dataset_name}{suffix}-visibility.json')
    with open(stats_path, 'w') as f:
        json.dump(
            dict(
                fovy_scale=FLAGS.fovy_scale,
                hide_lock_colors=FLAGS.hide_lock_colors,
                visibility_every=FLAGS.visibility_every,
                **stats,
            ),
            f,
            indent=2,
        )
    print(f'Saved: {stats_path}')


if __name__ == '__main__':
    app.run(main)
