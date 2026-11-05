"""Extract one array of an .npz file into an uncompressed .npy file, so that it can be memory-mapped.

The array is streamed in chunks, so this does not need memory for the whole array.

Usage:
    python scripts/npz_to_npy.py --path=~/.ogbench/data/visual-cube-single-play-v0-fovy0.35.npz --key=observations
"""

import os
import zipfile

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('path', None, 'Path to the .npz file.')
flags.DEFINE_string('key', 'observations', 'Array to extract.')
flags.DEFINE_integer('chunk_bytes', 256 * 2**20, 'Bytes to copy at a time.')


def npy_path(npz_path, key):
    """Path of the extracted array: `<name>-<key>.npy` next to `<name>.npz`."""
    return npz_path.removesuffix('.npz') + f'-{key}.npy'


def main(_):
    path = os.path.expanduser(FLAGS.path)
    out_path = npy_path(path, FLAGS.key)
    with zipfile.ZipFile(path) as zf, zf.open(f'{FLAGS.key}.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        assert not fortran_order
        out = np.lib.format.open_memmap(out_path + '.tmp', mode='w+', dtype=dtype, shape=shape)
        flat = out.reshape(-1).view(np.uint8)
        offset = 0
        while offset < flat.size:
            data = f.read(min(FLAGS.chunk_bytes, flat.size - offset))
            flat[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
            offset += len(data)
        out.flush()
        del out, flat
    os.replace(out_path + '.tmp', out_path)
    print(f'Saved: {out_path} {shape} {dtype}')


if __name__ == '__main__':
    flags.mark_flag_as_required('path')
    app.run(main)
