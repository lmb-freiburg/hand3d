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
import os

from utils.general import *
from utils.canonical_trafo import *
from utils.relative_trafo import *

ops = NetworkOps


class PosePriorNetwork(object):
    """ Network containing different variants for lifting 2D predictions into 3D. """
    def __init__(self, variant):
        self.num_kp = 21
        self.variant = variant

    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        import pickle
        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def inference(self, scoremap, hand_side, evaluation):
        """ Infere 3D coordinates from 2D scoremaps. """
        scoremap_pooled = tf.nn.avg_pool(scoremap, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        coord3d, R = None, None
        if self.variant == 'direct':
            coord_xyz_rel_normed = self._inference_pose3d(scoremap_pooled, hand_side, evaluation, train=True)
            coord3d = coord_xyz_rel_normed
        elif self.variant == 'bottleneck':
            coord_xyz_rel_normed = self._inference_pose3d(scoremap_pooled, hand_side, evaluation, train=True, bottleneck=True)
            coord3d = coord_xyz_rel_normed
        elif (self.variant == 'local') or (self.variant == 'local_w_xyz_loss'):
            coord_xyz_rel_loc = self._inference_pose3d(scoremap_pooled, hand_side, evaluation, train=True)
            coord3d = coord_xyz_rel_loc

            # assemble to real coords
            coord_xyz_rel_normed = bone_rel_trafo_inv(coord_xyz_rel_loc)
        elif self.variant == 'proposed':
            # infer coordinates in the canonical frame
            coord_can = self._inference_pose3d(scoremap_pooled, hand_side, evaluation, train=True)
            coord3d = coord_can

            # infer viewpoint
            rot_mat = self._inference_viewpoint(scoremap_pooled, hand_side, evaluation, train=True)
            R = rot_mat

            # flip hand according to hand side
            cond_right = tf.equal(tf.argmax(hand_side, 1), 1)
            cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]), [1, self.num_kp, 3])
            coord_xyz_can_flip = self._flip_right_hand(coord_can, cond_right_all)

            # rotate view back
            coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, rot_mat)
        else:
            assert 0, "Unknown variant."

        return coord_xyz_rel_normed, coord3d, R

    def _inference_pose3d(self, keypoints_scoremap, hand_side, evaluation, train=False, bottleneck=False):
        """ Inference of canonical coordinates. """
        with tf.variable_scope('PosePrior'):
            # use encoding to detect relative, normed 3d coords
            x = keypoints_scoremap  # this is 28x28x21
            s = x.get_shape().as_list()
            out_chan_list = [32, 64, 128]
            for i, out_chan in enumerate(out_chan_list):
                x = ops.conv_relu(x, 'conv_pose_%d_1' % i, kernel_size=3, stride=1, out_chan=out_chan, trainable=train)
                x = ops.conv_relu(x, 'conv_pose_%d_2' % i, kernel_size=3, stride=2, out_chan=out_chan, trainable=train) # in the end this will be 4x4xC

            # Estimate relative 3D coordinates
            out_chan_list = [512, 512]
            x = tf.reshape(x, [s[0], -1])
            x = tf.concat([x, hand_side], 1)
            for i, out_chan in enumerate(out_chan_list):
                x = ops.fully_connected_relu(x, 'fc_rel%d' % i, out_chan=out_chan, trainable=train)
                x = ops.dropout(x, 0.8, evaluation)
            if bottleneck:
                x = ops.fully_connected(x, 'fc_bottleneck', out_chan=30)
            coord_xyz_rel = ops.fully_connected(x, 'fc_xyz', out_chan=self.num_kp*3, trainable=train)

            # reshape stuff
            coord_xyz_rel = tf.reshape(coord_xyz_rel, [s[0], self.num_kp, 3])

            return coord_xyz_rel

    def _inference_viewpoint(self, keypoints_scoremap, hand_side, evaluation, train=False):
        """ Inference of the viewpoint. """
        with tf.variable_scope('ViewpointNet'):
            # estimate rotation
            ux, uy, uz = self._rotation_estimation(keypoints_scoremap, hand_side, evaluation, train=train)

            # assemble rotation matrix
            rot_mat = self._get_rot_mat(ux, uy, uz)

            return rot_mat

    @staticmethod
    def _rotation_estimation(scoremap2d, hand_side, evaluation, train=False):
        """ Estimates the rotation from canonical coords to realworld xyz. """
        # conv down scoremap to some reasonable length
        x = tf.concat([scoremap2d], 3)
        s = x.get_shape().as_list()
        out_chan_list = [64, 128, 256]
        for i, out_chan in enumerate(out_chan_list):
            x = ops.conv_relu(x, 'conv_vp_%d_1' % i, kernel_size=3, stride=1, out_chan=out_chan, trainable=train)
            x = ops.conv_relu(x, 'conv_vp_%d_2' % i, kernel_size=3, stride=2, out_chan=out_chan, trainable=train) # in the end this will be 4x4x128

        # flatten
        x = tf.reshape(x, [s[0], -1])  # this is Bx2048
        x = tf.concat([x, hand_side], 1)

        # Estimate Viewpoint --> 3 params
        out_chan_list = [256, 128]
        for i, out_chan in enumerate(out_chan_list):
            x = ops.fully_connected_relu(x, 'fc_vp%d' % i, out_chan=out_chan, trainable=train)
            x = ops.dropout(x, 0.75, evaluation)

        ux = ops.fully_connected(x, 'fc_vp_ux', out_chan=1, trainable=train)
        uy = ops.fully_connected(x, 'fc_vp_uy', out_chan=1, trainable=train)
        uz = ops.fully_connected(x, 'fc_vp_uz', out_chan=1, trainable=train)
        return ux, uy, uz

    def _get_rot_mat(self, ux_b, uy_b, uz_b):
        """ Returns a rotation matrix from axis and (encoded) angle."""
        with tf.name_scope('get_rot_mat'):
            u_norm = tf.sqrt(tf.square(ux_b) + tf.square(uy_b) + tf.square(uz_b) + 1e-8)
            theta = u_norm

            # some tmp vars
            st_b = tf.sin(theta)
            ct_b = tf.cos(theta)
            one_ct_b = 1.0 - tf.cos(theta)

            st = st_b[:, 0]
            ct = ct_b[:, 0]
            one_ct = one_ct_b[:, 0]
            norm_fac = 1.0 / u_norm[:, 0]
            ux = ux_b[:, 0] * norm_fac
            uy = uy_b[:, 0] * norm_fac
            uz = uz_b[:, 0] * norm_fac

            trafo_matrix = self._stitch_mat_from_vecs([ct+ux*ux*one_ct, ux*uy*one_ct-uz*st, ux*uz*one_ct+uy*st,
                                                       uy*ux*one_ct+uz*st, ct+uy*uy*one_ct, uy*uz*one_ct-ux*st,
                                                       uz*ux*one_ct-uy*st, uz*uy*one_ct+ux*st, ct+uz*uz*one_ct])

            return trafo_matrix

    @staticmethod
    def _flip_right_hand(coords_xyz_canonical, cond_right):
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

    @staticmethod
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
