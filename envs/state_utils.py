"""Privileged state vectors for OGBench manipulation environments.

A state is the observation `compute_observation()` returns in a state-based environment ('states' ob_type), with one
change: 'gripper_contact' is replaced with a binary indicator of geometric contact of the right gripper pad. The
original value is a contact force norm that depends on the control signal, which the datasets do not store, so it
cannot be reconstructed from qpos/qvel. The indicator agrees with `gripper_contact > 0` on 99.98% of the
cube-single-play-v0 transitions we checked.

Datasets store qpos/qvel after `mj_step`, but `mj_step` computes body positions and contacts before integrating qpos.
So the stored effector pose and contacts belong to the state one physics timestep earlier. `reconstruct_state` undoes
this by integrating qpos backwards by one timestep for those quantities (except at the first step of an episode,
which comes from `reset` and has no lag).
"""

import contextlib

import mujoco
import numpy as np

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
        if not is_initial:
            # Effector pose and contacts from the state one physics timestep earlier.
            prev_qpos = qpos.copy()
            mujoco.mj_integratePos(u._model, prev_qpos, qvel, -u._model.opt.timestep)
            _set_state(u, prev_qpos, qvel, button_states)
            state[EFFECTOR_DIMS] = u.compute_observation()[EFFECTOR_DIMS]
    state[GRIPPER_CONTACT_DIM] = _pad_touch(u)
    return state
