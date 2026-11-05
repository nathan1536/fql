from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@jax.jit
def concatenate_frames(*frames):
    """Concatenate frames along the last axis (on the accelerator: much faster than NumPy for few channels)."""
    return jnp.concatenate(frames, axis=-1)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class IndexedArray:
    """Read-only view `base[idxs]` that is resolved lazily on indexing.

    This lets 'observations' and 'next_observations' share one copy of the underlying data, which matters for large
    image datasets. Indexing with `arr[i]` returns exactly the same values as `base[idxs][i]`.
    """

    def __init__(self, base, idxs):
        self.base = base
        self.idxs = idxs

    def __getitem__(self, key):
        return self.base[self.idxs[key]]

    def __len__(self):
        return len(self.idxs)

    def __array__(self, dtype=None, copy=None):
        arr = self.base[self.idxs]
        return arr if dtype is None else arr.astype(dtype)

    @property
    def shape(self):
        return (len(self.idxs), *self.base.shape[1:])

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def ndim(self):
        return self.base.ndim

    def setflags(self, write):
        self.base.setflags(write=write)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        # Separate critic observations (e.g., from another camera), if any, are stacked and augmented the same way.
        ob_keys = [('observations', 'next_observations')]
        if 'critic_observations' in self:
            ob_keys.append(('critic_observations', 'next_critic_observations'))
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            for ob_key, next_ob_key in ob_keys:
                obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
                next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
                for i in reversed(range(self.frame_stack)):
                    # Use the initial state if the index is out of bounds.
                    cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                    obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self[ob_key]))
                    if i != self.frame_stack - 1:
                        next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self[ob_key]))
                next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self[next_ob_key]))

                batch[ob_key] = jax.tree_util.tree_map(lambda *args: np.asarray(concatenate_frames(*args)), *obs)
                batch[next_ob_key] = jax.tree_util.tree_map(
                    lambda *args: np.asarray(concatenate_frames(*args)), *next_obs
                )
        if self.p_aug is not None:
            # Apply random-crop image augmentation (the same crops for all observation keys).
            if np.random.rand() < self.p_aug:
                self.augment(batch, [k for keys in ob_keys for k in keys])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


class SegmentedArray:
    """Read-only concatenation of two arrays along the first axis, without copying them."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __getitem__(self, key):
        key = np.asarray(key)
        n = len(self.first)
        if key.ndim == 0:
            return self.first[key] if key < n else self.second[key - n]
        out = np.empty((*key.shape, *self.first.shape[1:]), dtype=self.first.dtype)
        in_first = key < n
        out[in_first] = self.first[key[in_first]]
        out[~in_first] = self.second[key[~in_first] - n]
        return out

    def __len__(self):
        return len(self.first) + len(self.second)

    @property
    def shape(self):
        return (len(self), *self.first.shape[1:])

    @property
    def dtype(self):
        return self.first.dtype

    @property
    def ndim(self):
        return self.first.ndim


class FrameReplayBuffer(ReplayBuffer):
    """Replay buffer that stores each image observation once.

    Transitions refer to frames by index, so `next_observations[t]` and `observations[t + 1]` share one frame, and the
    frames of the initial dataset are used in place instead of being copied. Observations are single (unstacked)
    frames; frame stacking happens in `sample`, as for the offline datasets. The buffer does not wrap around: it has
    room for exactly `num_transitions` new transitions. New frames can be stored in a file (`frames_path`) instead of
    in memory, so that memory use does not grow with the number of transitions.
    """

    @classmethod
    def create(cls, num_transitions, init_dataset=None, example_transition=None, frames_path=None):
        """Create a replay buffer.

        Args:
            num_transitions: Number of transitions to make room for (in addition to the initial dataset).
            init_dataset: Initial dataset whose 'observations' and 'next_observations' are `IndexedArray` views of
                the same frames (optional).
            example_transition: Example transition (dict); required if `init_dataset` is None.
            frames_path: File to store new frames in, through a memory map (optional; in memory by default).
        """
        ob_keys = ['observations', 'next_observations']
        if init_dataset is not None:
            obs, next_obs = init_dataset['observations'], init_dataset['next_observations']
            assert isinstance(obs, IndexedArray) and isinstance(next_obs, IndexedArray) and obs.base is next_obs.base
            init_frames, init_size = obs.base, len(obs)
            fields = {k: np.asarray(v) for k, v in init_dataset.items() if k not in ob_keys}
        else:
            ex_ob = np.asarray(example_transition['observations'])
            init_frames, init_size = np.zeros((0, *ex_ob.shape), dtype=ex_ob.dtype), 0
            fields = {k: np.asarray(v)[None] for k, v in example_transition.items() if k not in ob_keys}

        size = init_size + num_transitions
        # At most two new frames per transition (when it starts an episode). `np.zeros` only takes memory once written,
        # and the file of `np.memmap` is sparse, so unused frames take neither memory nor disk space.
        new_frames_shape = (2 * num_transitions, *init_frames.shape[1:])
        if frames_path is None:
            new_frames = np.zeros(new_frames_shape, dtype=init_frames.dtype)
        else:
            new_frames = np.memmap(frames_path, dtype=init_frames.dtype, mode='w+', shape=new_frames_shape)
        frames = SegmentedArray(init_frames, new_frames)

        ob_frame_idxs = np.zeros(size, dtype=np.int64)
        next_ob_frame_idxs = np.zeros(size, dtype=np.int64)
        data = dict(
            observations=IndexedArray(frames, ob_frame_idxs),
            next_observations=IndexedArray(frames, next_ob_frame_idxs),
        )
        if init_dataset is not None:
            ob_frame_idxs[:init_size] = obs.idxs
            next_ob_frame_idxs[:init_size] = next_obs.idxs
        for k, v in fields.items():
            data[k] = np.zeros((size, *v.shape[1:]), dtype=v.dtype)
            if init_dataset is not None:
                data[k][:init_size] = v

        buffer = cls(data)
        buffer.size = buffer.pointer = init_size
        buffer.new_frames = new_frames
        buffer.num_init_frames = len(init_frames)
        buffer.num_new_frames = 0
        return buffer

    def add_frame(self, frame):
        """Store a frame and return its index."""
        assert self.num_new_frames < len(self.new_frames), 'FrameReplayBuffer has no room for more frames.'
        self.new_frames[self.num_new_frames] = frame
        self.num_new_frames += 1
        return self.num_init_frames + self.num_new_frames - 1

    def add_transition(self, transition):
        """Add a transition. Its observations must be single frames, as in the datasets."""
        assert self.pointer < self.max_size, 'FrameReplayBuffer is full.'
        i = self.pointer
        ob_frame_idxs = self['observations'].idxs
        next_ob_frame_idxs = self['next_observations'].idxs

        if i == 0 or self['terminals'][i - 1] > 0:
            # First transition of an episode.
            ob_frame_idxs[i] = self.add_frame(transition['observations'])
            if i > 0:
                self.initial_locs = np.append(self.initial_locs, i)
        else:
            # The observation is the previous transition's next observation.
            ob_frame_idxs[i] = next_ob_frame_idxs[i - 1]
            assert np.array_equal(self['observations'][i], transition['observations']), 'Observation mismatch.'
        next_ob_frame_idxs[i] = self.add_frame(transition['next_observations'])

        for k, v in transition.items():
            if k not in ['observations', 'next_observations']:
                self[k][i] = v

        self.pointer += 1
        self.size = self.pointer
