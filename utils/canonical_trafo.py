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
import tensorflow as tf


def atan2(y, x):
    """ My implementation of atan2 in tensorflow.  Returns in -pi .. pi."""
    tan = tf.atan(y / (x + 1e-8))  # this returns in -pi/2 .. pi/2

    one_map = tf.ones_like(tan)

    # correct quadrant error
    correction = tf.where(tf.less(x + 1e-8, 0.0), 3.141592653589793*one_map, 0.0*one_map)
    tan_c = tan + correction  # this returns in -pi/2 .. 3pi/2

    # bring to positive values
    correction = tf.where(tf.less(tan_c, 0.0), 2*3.141592653589793*one_map, 0.0*one_map)
    tan_zero_2pi = tan_c + correction  # this returns in 0 .. 2pi

    # make symmetric
    correction = tf.where(tf.greater(tan_zero_2pi, 3.141592653589793), -2*3.141592653589793*one_map, 0.0*one_map)
    tan_final = tan_zero_2pi + correction  # this returns in -pi .. pi
    return tan_final


def _stitch_mat_from_vecs(vector_list):
    """ Stitches a given list of vectors into a 3x3 matrix.

        Input:
            vector_list: list of 9 tensors, which will be stitched into a matrix. list contains matrix elements
                in a row-first fashion (m11, m12, m13, m21, m22, m23, m31, m32, m33). Length of the vectors has
                to be the same, because it is interpreted as batch dimension.
    """

    assert len(vector_list) == 9, "There have to be exactly 9 tensors in vector_list."
    batch_size = vector_list[0].get_shape().as_list()[0]
    vector_list = [tf.reshape(x, [1, batch_size]) for x in vector_list]

    trafo_matrix = tf.dynamic_stitch([[0], [1], [2],
                                      [3], [4], [5],
                                      [6], [7], [8]], vector_list)

    trafo_matrix = tf.reshape(trafo_matrix, [3, 3, batch_size])
    trafo_matrix = tf.transpose(trafo_matrix, [2, 0, 1])

    return trafo_matrix


def _get_rot_mat_x(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec,
                                          zero_vec, tf.cos(angle), tf.sin(angle),
                                          zero_vec, -tf.sin(angle), tf.cos(angle)])
    return trafo_matrix


def _get_rot_mat_y(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([tf.cos(angle), zero_vec, -tf.sin(angle),
                                          zero_vec, one_vec, zero_vec,
                                          tf.sin(angle), zero_vec, tf.cos(angle)])
    return trafo_matrix


def _get_rot_mat_z(angle):
    """ Returns a 3D rotation matrix. """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([tf.cos(angle), tf.sin(angle), zero_vec,
                                          -tf.sin(angle), tf.cos(angle), zero_vec,
                                          zero_vec, zero_vec, one_vec])
    return trafo_matrix


def canonical_trafo(coords_xyz):
    """ Transforms the given real xyz coordinates into some canonical frame.
        Within that frame the hands of all frames are nicely aligned, which
        should help the network to learn reasonable shape priors.

        Inputs:
            coords_xyz: BxNx3 matrix, containing the coordinates for each of the N keypoints
    """
    with tf.variable_scope('canonical-trafo'):
        coords_xyz = tf.reshape(coords_xyz, [-1, 21, 3])

        ROOT_NODE_ID = 0  # Node that will be at 0/0/0: 0=palm keypoint (root)
        ALIGN_NODE_ID = 12  # Node that will be at 0/-D/0: 12=beginning of middle finger
        ROT_NODE_ID = 20  # Node that will be at z=0, x>0; 20: Beginning of pinky

        # 1. Translate the whole set s.t. the root kp is located in the origin
        trans = tf.expand_dims(coords_xyz[:, ROOT_NODE_ID, :], 1)
        coords_xyz_t = coords_xyz - trans

        # 2. Rotate and scale keypoints such that the root bone is of unit length and aligned with the y axis
        p = coords_xyz_t[:, ALIGN_NODE_ID, :]  # thats the point we want to put on (0/1/0)

        # Rotate point into the yz-plane
        alpha = atan2(p[:, 0], p[:, 1])
        rot_mat = _get_rot_mat_z(alpha)
        coords_xyz_t_r1 = tf.matmul(coords_xyz_t, rot_mat)
        total_rot_mat = rot_mat

        # Rotate point within the yz-plane onto the xy-plane
        p = coords_xyz_t_r1[:, ALIGN_NODE_ID, :]
        beta = -atan2(p[:, 2], p[:, 1])
        rot_mat = _get_rot_mat_x(beta + 3.141592653589793)
        coords_xyz_t_r2 = tf.matmul(coords_xyz_t_r1, rot_mat)
        total_rot_mat = tf.matmul(total_rot_mat, rot_mat)

        # 3. Rotate keypoints such that rotation along the y-axis is defined
        p = coords_xyz_t_r2[:, ROT_NODE_ID, :]
        gamma = atan2(p[:, 2], p[:, 0])
        rot_mat = _get_rot_mat_y(gamma)
        coords_xyz_normed = tf.matmul(coords_xyz_t_r2, rot_mat)
        total_rot_mat = tf.matmul(total_rot_mat, rot_mat)

        return coords_xyz_normed, total_rot_mat


def flip_right_hand(coords_xyz_canonical, cond_right):
    """ Flips the given canonical coordinates, when cond_right is true. Returns coords unchanged otherwise.
        The returned coordinates represent those of a left hand.

        Inputs:
            coords_xyz_canonical: Nx3 matrix, containing the coordinates for each of the N keypoints
    """
    with tf.variable_scope('flip-right-hand'):
        expanded = False
        s = coords_xyz_canonical.get_shape().as_list()
        if len(s) == 2:
            coords_xyz_canonical = tf.expand_dims(coords_xyz_canonical, 0)
            cond_right = tf.expand_dims(cond_right, 0)
            expanded = True

        # mirror along y axis
        coords_xyz_canonical_mirrored = tf.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], -coords_xyz_canonical[:, :, 2]], -1)

        # select mirrored in case it was a right hand
        coords_xyz_canonical_left = tf.where(cond_right, coords_xyz_canonical_mirrored, coords_xyz_canonical)

        if expanded:
            coords_xyz_canonical_left = tf.squeeze(coords_xyz_canonical_left, [0])

        return coords_xyz_canonical_left