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

ops = NetworkOps


class ColorHandPose3DNetwork(object):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def __init__(self):
        self.crop_size = 256
        self.num_kp = 21

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

        if weight_files is None:
            weight_files = ['./weights/handsegnet-rhd.pickle', './weights/posenet3d-rhd-stb-slr-finetuned.pickle']

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

    def inference(self, image, hand_side, evaluation):
        """ Full pipeline: HandSegNet + PoseNet + PosePrior.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                hand_side: [B, 2] tf.float32 tensor, One hot encoding if the image is showing left or right side
                evaluation: [] tf.bool tensor, True while evaluation false during training (controls dropout)

            Outputs:
                hand_scoremap: [B, H, W, 2] tf.float32 tensor, Scores for background and hand class
                image_crop: [B, 256, 256, 3] tf.float32 tensor, Hand cropped input image
                scale_crop: [B, 1] tf.float32 tensor, Scaling between input image and image_crop
                center: [B, 1] tf.float32 tensor, Center of image_crop wrt to image
                keypoints_scoremap: [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
                keypoint_coord3d: [B, 21, 3] tf.float32 tensor, Normalized 3D coordinates
        """
        # use network for hand segmentation for detection
        hand_scoremap = self.inference_detection(image)
        hand_scoremap = hand_scoremap[-1]

        # Intermediate data processing
        hand_mask = single_obj_scoremap(hand_scoremap)
        center, _, crop_size_best = calc_center_bb(hand_mask)
        crop_size_best *= 1.25
        scale_crop = tf.minimum(tf.maximum(self.crop_size / crop_size_best, 0.25), 5.0)
        image_crop = crop_image_from_xy(image, center, self.crop_size, scale=scale_crop)

        # detect keypoints in 2D
        keypoints_scoremap = self.inference_pose2d(image_crop)
        keypoints_scoremap = keypoints_scoremap[-1]

        # estimate most likely 3D pose
        keypoint_coord3d = self._inference_pose3d(keypoints_scoremap, hand_side, evaluation)

        # upsample keypoint scoremap
        s = image_crop.get_shape().as_list()
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))

        return hand_scoremap, image_crop, scale_crop, center, keypoints_scoremap, keypoint_coord3d

    def inference2d(self, image):
        """ Only 2D part of the pipeline: HandSegNet + PoseNet.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted

            Outputs:
                image_crop: [B, 256, 256, 3] tf.float32 tensor, Hand cropped input image
                scale_crop: [B, 1] tf.float32 tensor, Scaling between input image and image_crop
                center: [B, 1] tf.float32 tensor, Center of image_crop wrt to image
                keypoints_scoremap: [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        # use network for hand segmentation for detection
        hand_scoremap = self.inference_detection(image)
        hand_scoremap = hand_scoremap[-1]

        # Intermediate data processing
        hand_mask = single_obj_scoremap(hand_scoremap)
        center, _, crop_size_best = calc_center_bb(hand_mask)
        crop_size_best *= 1.25
        scale_crop = tf.minimum(tf.maximum(self.crop_size / crop_size_best, 0.25), 5.0)
        image_crop = crop_image_from_xy(image, center, self.crop_size, scale=scale_crop)

        # detect keypoints in 2D
        s = image_crop.get_shape().as_list()
        keypoints_scoremap = self.inference_pose2d(image_crop)
        keypoints_scoremap = keypoints_scoremap[-1]
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))
        return keypoints_scoremap, image_crop, scale_crop, center

    @staticmethod
    def inference_detection(image, train=False):
        """ HandSegNet: Detects the hand in the input image by segmenting it.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 2] tf.float32 tensor, Scores for the hand segmentation classes
        """
        with tf.variable_scope('HandSegNet'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 4]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv5_1', kernel_size=3, stride=1, out_chan=512, trainable=train)
            encoding = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv6_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv6_2', kernel_size=1, stride=1, out_chan=2, trainable=train)
            scoremap_list.append(scoremap)

            # upsample to full size
            s = image.get_shape().as_list()
            scoremap_list_large = [tf.image.resize_images(x, (s[1], s[2])) for x in scoremap_list]

        return scoremap_list_large

    def inference_pose2d(self, image_crop, train=False):
        """ PoseNet: Given an image it detects the 2D hand keypoints.
            The image should already contain a rather tightly cropped hand.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        with tf.variable_scope('PoseNet2D'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image_crop
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv_relu(x, 'conv4_7', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv5_2', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            for pass_id in range(num_recurrent_units):
                x = tf.concat([scoremap_list[-1], encoding], 3)
                for rec_id in range(layers_per_recurrent_unit):
                    x = ops.conv_relu(x, 'conv%d_%d' % (pass_id+6, rec_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv%d_6' % (pass_id+6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                scoremap = ops.conv(x, 'conv%d_7' % (pass_id+6), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                scoremap_list.append(scoremap)

            scoremap_list_large = scoremap_list

        return scoremap_list_large

    def _inference_pose3d(self, keypoints_scoremap, hand_side, evaluation, train=False):
        """ PosePrior + Viewpoint: Estimates the most likely normalized 3D pose given 2D detections and hand side.

            Inputs:
                keypoints_scoremap: [B, 32, 32, 21] tf.float32 tensor, Scores for the hand keypoints
                hand_side: [B, 2] tf.float32 tensor, One hot encoding if the image is showing left or right side
                evaluation: [] tf.bool tensor, True while evaluation false during training (controls dropout)
                train: bool, True in case weights should be trainable

            Outputs:
                coord_xyz_rel_normed: [B, 21, 3] tf.float32 tensor, Normalized 3D coordinates
        """
        # infer coordinates in the canonical frame
        coord_can = self._inference_pose3d_can(keypoints_scoremap, hand_side, evaluation, train=train)

        # infer viewpoint
        rot_mat = self._inference_viewpoint(keypoints_scoremap, hand_side, evaluation, train=train)

        # flip hand according to hand side
        cond_right = tf.equal(tf.argmax(hand_side, 1), 1)
        cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]), [1, self.num_kp, 3])
        coord_xyz_can_flip = self._flip_right_hand(coord_can, cond_right_all)

        # rotate view back
        coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, rot_mat)

        return coord_xyz_rel_normed

    def _inference_pose3d_can(self, keypoints_scoremap, hand_side, evaluation, train=False):
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
