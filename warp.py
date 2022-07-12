import tensorflow as tf
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
import matplotlib.tri as mtri

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def tf_batch_map_coordinates(_input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(_input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.math.floor(coords), 'int32')
    coords_rb = tf.cast(tf.math.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(__input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
        ], axis=-1)
        vals = tf.gather_nd(__input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords, __input.shape[3]))
        return vals

    vals_lt = _get_vals_by_coords(_input, coords_lt)
    vals_rb = _get_vals_by_coords(_input, coords_rb)
    vals_lb = _get_vals_by_coords(_input, coords_lb)
    vals_rt = _get_vals_by_coords(_input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    offset_0 =coords_offset_lt[..., 0]
    offset_1 =coords_offset_lt[..., 1]
    offset_0 = tf.reshape(offset_0, [offset_0.shape[0], offset_0.shape[1], 1])
    offset_1 = tf.reshape(offset_1, [offset_1.shape[0], offset_1.shape[1], 1])
    vals_t = vals_lt + (vals_rt - vals_lt) * offset_0
    vals_b = vals_lb + (vals_rb - vals_lb) * offset_0
    mapped_vals = vals_t + (vals_b - vals_t) * offset_1

    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_batch_map_offsets(_input, offsets, order=1):
    input_size = _input.shape[1]
    offsets = tf.image.resize(offsets, [input_size, input_size]) * input_size
    offsets = offsets[:,:,:,0:2]

    """Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tf.shape(_input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(_input, coords)
    mapped_vals = tf.reshape(mapped_vals, (batch_size, input_size, input_size, -1))

    return mapped_vals

def generate_offset_map_batch(source, target, img_size):
    offsetmap_batch = []
    for _source, _target in zip(tf.unstack(source), tf.unstack(target)):
        offsetmap = generate_offset_map(_source, _target, img_size)
        offsetmap_batch.append(offsetmap)
    return tf.stack(offsetmap_batch, axis=0)

def generate_offset_map(source, target, img_size):
    anchor_pts = [[0,0],[0,255],[255,0],[255,255],
                  [0,127],[127,0],[255,127],[127,255],
                  [0,63],[0,191],[255,63],[255,191],
                  [63,0],[191,0],[63,255],[191,255]]
    anchor_pts = np.asarray(anchor_pts)/ 255
    xi, yi = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    _source = np.concatenate([source, anchor_pts], axis=0).astype(np.float32)
    _target = np.concatenate([target, anchor_pts], axis=0).astype(np.float32)
    _offset = _source - _target

    # interp2d
    _triang = mtri.Triangulation(_target[:,0], _target[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, _offset[:,0])
    _interpy = mtri.LinearTriInterpolator(_triang, _offset[:,1])
    _offsetmapx = _interpx(xi, yi)
    _offsetmapy = _interpy(xi, yi)

    offsetmap = np.stack([_offsetmapy, _offsetmapx, _offsetmapx*0], axis=2)
    return offsetmap

def generate_uv_map(source, uv, img_size):
    xi, yi = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))

    # interp2d
    _triang = mtri.Triangulation(source[:,0], source[:,1]) 
    _interpz = mtri.LinearTriInterpolator(_triang, uv[:,2])
    _offsetmapz = _interpz(xi, yi)

    offsetmap = np.reshape(_offsetmapz,(img_size,img_size,1))
    offsetmap = np.nan_to_num(offsetmap)  
    return offsetmap

