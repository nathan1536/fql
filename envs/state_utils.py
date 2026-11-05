"""Privileged state vectors for OGBench manipulation environments.

A state is the observation `compute_observation()` returns in a state-based environment ('states' ob_type), with one
change: 'gripper_contact' is replaced with a binary indicator of geometric contact of the right gripper pad. The
original value is a contact force norm that depends on the control signal, which the datasets do not store, so it
cannot be reconstructed from qpos/qvel. The indicator agrees with `gripper_contact > 0` on 99.98% of the
cube-single-play-v0 transitions we checked.

Datasets store qpos/qvel after `mj_step`, but `mj_step` computes body positions and contacts before integrating qpos.
So in the cube environments, the stored effector pose and contacts belong to the state one physics timestep earlier.
`reconstruct_state` undoes this by integrating qpos backwards by one timestep for those quantities (except at the
first step of an episode, which comes from `reset` and has no lag). The scene and puzzle environments have no lag:
they call `_apply_button_states` (which ends with `mj_forward`) after every step.
"""

import contextlib

import mujoco
import numpy as np
from ogbench.manipspace import lie

EFFECTOR_DIMS = slice(12, 17)  # effector_pos (3), cos(effector_yaw), sin(effector_yaw).
GRIPPER_CONTACT_DIM = 18
CONTACT_DIST = 1e-3


@contextlib.contextmanager
def _states_ob_type(u):
    """Temporarily make `compute_observation()` return states (also for pixel-based environments)."""
    ob_type = u._ob_type
    u._ob_type = 'states'
    try:
        yield
    finally:
        u._ob_type = ob_type


def _pad_touch(u):
    """Return 1.0 if the right gripper pad is in geometric contact with anything, else 0.0."""
    m, d = u._model, u._data
    pad = m.body('ur5e/robotiq/right_pad').id
    for c in d.contact[: d.ncon]:
        if pad in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and c.dist < CONTACT_DIST:
            return 1.0
    return 0.0


def _set_state(u, qpos, qvel, button_states):
    if button_states is None:
        u.set_state(qpos, qvel)
    else:
        u.set_state(qpos, qvel, button_states)


def _has_lag(u):
    """Whether body positions and contacts after `step` are from one physics timestep earlier (see above)."""
    return not hasattr(u, '_apply_button_states')


def compute_live_state(env):
    """Return the state of the environment right after `reset` or `step`."""
    u = env.unwrapped
    with _states_ob_type(u):
        state = u.compute_observation()
    state[GRIPPER_CONTACT_DIM] = _pad_touch(u)
    return state


def reconstruct_state(env, qpos, qvel, button_states=None, is_initial=False):
    """Reconstruct the state from qpos/qvel (and button states) stored in a dataset.

    Args:
        env: OGBench manipulation environment. Its simulation state is overwritten.
        qpos: Stored qpos.
        qvel: Stored qvel.
        button_states: Stored button states (scene and puzzle environments).
        is_initial: Whether this is the first step of an episode.
    """
    u = env.unwrapped
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    with _states_ob_type(u):
        _set_state(u, qpos, qvel, button_states)
        state = u.compute_observation()
        if not is_initial and _has_lag(u):
            # Effector pose and contacts from the state one physics timestep earlier.
            prev_qpos = qpos.copy()
            mujoco.mj_integratePos(u._model, prev_qpos, qvel, -u._model.opt.timestep)
            _set_state(u, prev_qpos, qvel, button_states)
            state[EFFECTOR_DIMS] = u.compute_observation()[EFFECTOR_DIMS]
    state[GRIPPER_CONTACT_DIM] = _pad_touch(u)
    return state


def set_rendered_state(env, qpos, qvel, button_states=None, is_initial=False):
    """Set the simulation to the state that a stored pixel observation shows.

    Rendering uses body positions, which (in environments with lag, see above) are from one physics timestep before
    the stored qpos/qvel, except at the first step of an episode.
    """
    u = env.unwrapped
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if not is_initial and _has_lag(u):
        qpos = qpos.copy()
        mujoco.mj_integratePos(u._model, qpos, qvel, -u._model.opt.timestep)
    _set_state(u, qpos, qvel, button_states)


def singular_yaw_mask(env, qpos, shape, max_rzz=1e-2):
    """Mask of the cube yaw dimensions (cos, sin) at rows where the cube's z-axis is nearly horizontal.

    The yaw is ill-defined there, so tiny differences in the quaternion (e.g., float32 rounding) can change it
    arbitrarily. Comparisons of states should skip these entries; the quaternion itself is still compared.

    Args:
        env: OGBench manipulation environment.
        qpos: Array of qpos (rows).
        shape: Shape of the states (rows, state dimension).
        max_rzz: Threshold on |R_zz| of the cube rotation below which the yaw is considered ill-defined.
    """
    mask = np.zeros(shape, dtype=bool)
    for i in range(getattr(env.unwrapped, '_num_cubes', 0)):
        quat_start = 14 + 7 * i + 3  # qpos: arm and gripper joints (14), then (xyz, wxyz) per cube.
        quats = np.asarray(qpos, np.float64)[:, quat_start : quat_start + 4]
        rzz = np.array([lie.SO3(wxyz=q).as_matrix()[2, 2] for q in quats])
        yaw_dim = 19 + 9 * i + 7  # State: proprio (19), then (xyz, wxyz, cos, sin) per cube.
        mask[np.abs(rzz) < max_rzz, yaw_dim : yaw_dim + 2] = True
    return mask
