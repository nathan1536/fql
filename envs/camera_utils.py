"""Rendering changes for pixel-based OGBench manipulation environments (occlusion tests).

Only the rendering changes; the physics is untouched.
- Camera: narrowing the field of view of the pixel camera ('front_pixels') zooms in, so that objects (and the gripper)
  leave the image part of the time.
- Lock colors: in the scene environments, the buttons and the drawer and window handles are red when locked and white
  when unlocked. Rendering them in a fixed color removes the lock state from the images.
"""

import gymnasium
import numpy as np

PIXEL_CAMERA = 'front_pixels'
# Not the default geom color (0.5, 0.5, 0.5, 1): MuJoCo renders geoms with that color in their material's color.
LOCK_NEUTRAL_RGBA = np.array([0.6, 0.6, 0.6, 1.0])


def hide_lock_colors(env):
    """Render the buttons and the drawer and window handles of a scene environment in a fixed color.

    The environment sets these colors from the lock state whenever its state is set, so call this before each render.
    """
    u = env.unwrapped
    for ids in u._button_geom_ids_list:
        for gid in ids:
            u._model.geom_rgba[gid] = LOCK_NEUTRAL_RGBA
    u._model.material('drawer_handle').rgba = LOCK_NEUTRAL_RGBA
    u._model.material('window_handle').rgba = LOCK_NEUTRAL_RGBA


def camera_suffix(fovy_scale):
    """Suffix of the dataset files rendered with a changed camera (e.g., '-fovy0.35')."""
    return '' if fovy_scale is None else f'-fovy{fovy_scale:g}'


def set_camera(env, fovy_scale):
    """Scale the field of view of the pixel camera. Returns whether the camera changed.

    The environment must have been reset (so that its model exists). The default field of view is read from the
    compiled camera description once, so calling this again does not compound the scale.
    """
    u = env.unwrapped
    m = u._model
    cam = m.camera(PIXEL_CAMERA).id
    if not hasattr(u, '_default_pixel_fovy'):
        u._default_pixel_fovy = float(m.cam_fovy[cam])
    fovy = u._default_pixel_fovy * fovy_scale
    changed = float(m.cam_fovy[cam]) != fovy
    m.cam_fovy[cam] = fovy
    return changed


class CameraWrapper(gymnasium.Wrapper):
    """Environment wrapper that renders pixel observations with a scaled field of view."""

    def __init__(self, env, fovy_scale):
        super().__init__(env)
        self.fovy_scale = fovy_scale

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        if set_camera(self.env, self.fovy_scale):
            # The model is compiled during the first reset, after which its camera can change; re-render.
            ob = self.env.unwrapped.compute_observation()
        return ob, info
