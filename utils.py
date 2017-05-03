#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import math


class NetworkOps(object):
    """ Operations that are frequently used within networks. """
    neg_slope_of_relu = 0.01

    @classmethod
    def leaky_relu(cls, tensor, name='relu'):
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name=name)
        return out_tensor

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def max_pool(cls, bottom, name='pool'):
        pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def upconv(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            kernel_shape = [kernel_size, kernel_size, in_size[3], in_size[3]]
            strides = [1, stride, stride, 1]

            # conv
            kernel = cls.get_deconv_filter(kernel_shape, trainable)
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[2]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def get_deconv_filter(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @staticmethod
    def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
            weights_shape = [in_size[1], out_chan]

            # weight matrix
            weights = tf.get_variable('weights', weights_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer(), trainable=trainable)
            weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

            # bias
            biases = tf.get_variable('biases', [out_chan], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable)
            biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

            out_tensor = tf.matmul(in_tensor, weights) + biases
            return out_tensor

    @classmethod
    def fully_connected_relu(cls, in_tensor, layer_name, out_chan, trainable=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, trainable)
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name='out')
        return out_tensor

    @staticmethod
    def dropout(in_tensor, keep_prob, evaluation):
        """ Dropout: Each neuron is dropped independently. """
        with tf.variable_scope('dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=tensor_shape))
            return out_tensor

    @staticmethod
    def spatial_dropout(in_tensor, keep_prob, evaluation):
        """ Spatial dropout: Not each neuron is dropped independently, but feature map wise. """
        with tf.variable_scope('spatial_dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=[tensor_shape[0], 1, 1, tensor_shape[3]]))
            return out_tensor


def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image. When factor is not given does an central crop.

    Inputs:
        image: 4D tensor, [batch, height, width, channels] which will be cropped in height and width dimension
        crop_location: tensor, [batch, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 4D tensor, [batch, crop_size, crop_size, channels]
    """
    with tf.name_scope('crop_image_from_xy'):
        s = image.get_shape().as_list()
        assert len(s) == 4, "Image needs to be of shape [batch, width, height, channel]"
        scale = tf.reshape(scale, [-1])
        crop_location = tf.cast(crop_location, tf.float32)
        crop_location = tf.reshape(crop_location, [s[0], 2])
        crop_size = tf.cast(crop_size, tf.float32)

        crop_size_scaled = crop_size / scale
        y1 = crop_location[:, 0] - crop_size_scaled//2
        y2 = y1 + crop_size_scaled
        x1 = crop_location[:, 1] - crop_size_scaled//2
        x2 = x1 + crop_size_scaled
        y1 /= s[1]
        y2 /= s[1]
        x1 /= s[2]
        x2 /= s[2]
        boxes = tf.pack([y1, x1, y2, x2], -1)

        crop_size = tf.cast(tf.pack([crop_size, crop_size]), tf.int32)
        box_ind = tf.range(s[0])
        image_c = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes, box_ind, crop_size, name='crop')
        return image_c


def find_max_location(scoremap):
    """ Returns the coordinates of the given scoremap with maximum value. """
    with tf.variable_scope('find_max_location'):
        s = scoremap.get_shape().as_list()
        if len(s) == 4:
            scoremap = tf.squeeze(scoremap, [3])
        if len(s) == 2:
            scoremap = tf.expand_dims(scoremap, 0)

        s = scoremap.get_shape().as_list()
        assert len(s) == 3, "Scoremap must be 3D."
        assert (s[0] < s[1]) and (s[0] < s[2]), "Scoremap must be [Batch, Width, Height]"

        # my meshgrid
        x_range = tf.expand_dims(tf.range(s[1]), 1)
        y_range = tf.expand_dims(tf.range(s[2]), 0)
        X = tf.tile(x_range, [1, s[2]])
        Y = tf.tile(y_range, [s[1], 1])

        x_vec = tf.reshape(X, [-1])
        y_vec = tf.reshape(Y, [-1])
        scoremap_vec = tf.reshape(scoremap, [s[0], -1])
        max_ind_vec = tf.cast(tf.argmax(scoremap_vec, dimension=1), tf.int32)

        xy_loc = list()
        for i in range(s[0]):
            x_loc = tf.reshape(x_vec[max_ind_vec[i]], [1])
            y_loc = tf.reshape(y_vec[max_ind_vec[i]], [1])
            xy_loc.append(tf.concat(0, [x_loc, y_loc]))

        xy_loc = tf.pack(xy_loc, 0)
        return xy_loc


def single_obj_scoremap(scoremap):
    """ Applies my algorithm to figure out the most likely object from a given segmentation scoremap. """
    with tf.variable_scope('single_obj_scoremap'):
        filter_size = 21
        s = scoremap.get_shape().as_list()
        assert len(s) == 4, "Scoremap must be 4D."

        scoremap_softmax = tf.nn.softmax(scoremap)  #B, H, W, C --> normalizes across last dimension
        scoremap_fg = tf.reduce_max(scoremap_softmax[:, :, :, 1:], 3) # B, H, W
        detmap_fg = tf.round(scoremap_fg) # B, H, W

        # find maximum in the fg scoremap
        max_loc = find_max_location(scoremap_fg)

        # use maximum to start "growing" our objectmap
        objectmap_list = list()
        kernel_dil = tf.ones((filter_size, filter_size, 1)) / float(filter_size*filter_size)
        for i in range(s[0]):
            # create initial objectmap (put a one at the maximum)
            sparse_ind = tf.reshape(max_loc[i, :], [1, 2])  # reshape that its one point with 2dim)
            objectmap = tf.sparse_to_dense(sparse_ind, [s[1], s[2]], 1.0)

            # grow the map by dilation and pixelwise and
            num_passes = max(s[1], s[2]) // (filter_size//2) # number of passes needes to make sure the map can spread over the whole image
            for j in range(num_passes):
                objectmap = tf.reshape(objectmap, [1, s[1], s[2], 1])
                objectmap_dil = tf.nn.dilation2d(objectmap, kernel_dil, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
                objectmap_dil = tf.reshape(objectmap_dil, [s[1], s[2]])
                objectmap = tf.round(tf.mul(detmap_fg[i, :, :], objectmap_dil))

            objectmap = tf.reshape(objectmap, [s[1], s[2], 1])
            objectmap_list.append(objectmap)

        objectmap = tf.pack(objectmap_list)

        return objectmap


def calc_center_bb(binary_class_mask):
    """ Returns the center of mass coordinates for the given binary_class_mask. """
    with tf.variable_scope('calc_center_bb'):
        binary_class_mask = tf.cast(binary_class_mask, tf.int32)
        binary_class_mask = tf.equal(binary_class_mask, 1)
        s = binary_class_mask.get_shape().as_list()
        if len(s) == 4:
            binary_class_mask = tf.squeeze(binary_class_mask, [3])

        s = binary_class_mask.get_shape().as_list()
        assert len(s) == 3, "binary_class_mask must be 3D."
        assert (s[0] < s[1]) and (s[0] < s[2]), "binary_class_mask must be [Batch, Width, Height]"

        # my meshgrid
        x_range = tf.expand_dims(tf.range(s[1]), 1)
        y_range = tf.expand_dims(tf.range(s[2]), 0)
        X = tf.tile(x_range, [1, s[2]])
        Y = tf.tile(y_range, [s[1], 1])

        bb_list = list()
        center_list = list()
        crop_size_list = list()
        for i in range(s[0]):
            X_masked = tf.cast(tf.boolean_mask(X, binary_class_mask[i, :, :]), tf.float32)
            Y_masked = tf.cast(tf.boolean_mask(Y, binary_class_mask[i, :, :]), tf.float32)

            x_min = tf.reduce_min(X_masked)
            x_max = tf.reduce_max(X_masked)
            y_min = tf.reduce_min(Y_masked)
            y_max = tf.reduce_max(Y_masked)

            start = tf.pack([x_min, y_min])
            end = tf.pack([x_max, y_max])
            bb = tf.pack([start, end], 1)
            bb_list.append(bb)

            center_x = 0.5*(x_max + x_min)
            center_y = 0.5*(y_max + y_min)
            center = tf.pack([center_x, center_y], 0)

            center = tf.cond(tf.reduce_all(tf.is_finite(center)), lambda: center,
                                  lambda: tf.constant([160.0, 160.0]))
            center.set_shape([2])
            center_list.append(center)

            crop_size_x = x_max - x_min
            crop_size_y = y_max - y_min
            crop_size = tf.expand_dims(tf.maximum(crop_size_x, crop_size_y), 0)
            crop_size = tf.cond(tf.reduce_all(tf.is_finite(crop_size)), lambda: crop_size,
                                  lambda: tf.constant([100.0]))
            crop_size.set_shape([1])
            crop_size_list.append(crop_size)

        bb = tf.pack(bb_list)
        center = tf.pack(center_list)
        crop_size = tf.pack(crop_size_list)

        return center, bb, crop_size


def detect_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
    return keypoint_coords


def trafo_coords(keypoints_crop_coords, centers, scale, crop_size):
    """ Transforms coords into global image coordinates. """
    keypoints_coords = np.copy(keypoints_crop_coords)

    keypoints_coords -= crop_size // 2

    keypoints_coords /= scale

    keypoints_coords += centers

    return keypoints_coords


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90., elev=90.)
