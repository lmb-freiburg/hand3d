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
import os

import tensorflow as tf

from utils.general import crop_image_from_xy
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.relative_trafo import bone_rel_trafo


class BinaryDbReaderSTB(object):
    """
        Reads data from the STB Dataset
    """
    def __init__(self, mode=None, batch_size=1, shuffle=True, use_wrist_coord=True, sigma=25.0, hand_crop=False,
                 random_crop_to_size=False,
                 hue_aug=False,
                 coord_uv_noise=False,
                 crop_center_noise=False, crop_scale_noise=False, crop_offset_noise=False,
                 scoremap_dropout=False):
        """ Inputs:
                mode: string, Indicates which binary file to read. Can be 'training' or 'evaluation'
                batch_size: int, Number of samples forming a batch
                shuffle: boolean, If true samples of binary file are shuffled while reading
                use_wrist_coord: boolean, When true keypoint #0 is the wrist, palm center otherwise
                hand_crop: boolean, When true calculates a tight hand crop using the gt keypoint annotations
                                    Updates/Sets the following output items: image_crop, crop_scale, scoremap, cam_mat, keypoint_uv21
                sigma: float, Size of the ground truth scoremaps
                random_crop_to_size: boolean, Takes randomly sampled crops from the image & the mask
                hue_aug: boolean, Random hue augmentation
                coord_uv_noise: boolean, Adds some gaussian noise on the gt uv coordinate
                crop_center_noise: boolean, Adds gaussian noise on the hand crop center location (all keypoints still lie within the crop)
                crop_scale_noise: boolean, Adds gaussian noise on the hand crop size
                crop_offset_noise: boolean, Offsets the hand crop center randomly (keypoints can lie outside the crop)
                scoremap_dropout: boolean, Randomly drop scoremap channels
        """
        self.num_samples = 0
        if mode == 'training':
            # self.path_to_db = './data/stb_train_shuffled.bin'
            self.num_samples = 30000
            assert 0, "This set is not for training!"
        elif mode == 'evaluation':
            self.path_to_db = './data/stb/stb_eval.bin'
            self.num_samples = 6000
        else:
            assert 0, "Unknown dataset mode."

        assert os.path.exists(self.path_to_db), "Could not find the binary data file!"

        # general parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.shuffle = shuffle
        self.use_wrist_coord = use_wrist_coord
        self.random_crop_to_size = random_crop_to_size
        self.random_crop_size = 256

        # data augmentation parameters
        self.hue_aug = hue_aug
        self.hue_aug_max = 0.1

        self.hand_crop = hand_crop
        self.coord_uv_noise = coord_uv_noise
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = crop_center_noise
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints

        self.crop_scale_noise = crop_scale_noise
        self.crop_offset_noise = crop_offset_noise
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        self.scoremap_dropout = scoremap_dropout
        self.scoremap_dropout_prob = 0.8

        # these are constants of the dataset and therefore must not be changed
        self.image_size = (480, 640)
        self.crop_size = 256
        self.num_kp = 21

    def get(self):
        """ Provides input data to the graph. """
        # calculate size of each record (this lists what is contained in the db and how many bytes are occupied)
        record_bytes = 0

        encoding_bytes = 4
        kp_xyz_entries = 3 * self.num_kp
        record_bytes += encoding_bytes*kp_xyz_entries

        encoding_bytes = 4
        kp_uv_entries = 2 * self.num_kp
        record_bytes += encoding_bytes*kp_uv_entries

        kp_vis_entries = self.num_kp
        record_bytes += encoding_bytes*kp_vis_entries

        image_bytes = self.image_size[0] * self.image_size[1] * 3
        record_bytes += image_bytes

        """ READ DATA ITEMS"""
        # Start reader
        reader = tf.FixedLengthRecordReader(header_bytes=0, record_bytes=record_bytes)
        _, value = reader.read(tf.train.string_input_producer([self.path_to_db]))

        # decode to floats
        bytes_read = 0
        data_dict = dict()
        record_bytes_float32 = tf.decode_raw(value, tf.float32)

        # 1. Read keypoint xyz
        keypoint_xyz21 = tf.reshape(tf.slice(record_bytes_float32, [bytes_read//4], [kp_xyz_entries]), [self.num_kp, 3])
        bytes_read += encoding_bytes*kp_xyz_entries
        keypoint_xyz21 /= 1000.0  # scale to meters
        keypoint_xyz21 = self.convert_kp(keypoint_xyz21)

        # calculate wrist coord
        if self.use_wrist_coord:
            wrist_xyz = keypoint_xyz21[16, :] + 2.0*(keypoint_xyz21[0, :] - keypoint_xyz21[16, :])
            keypoint_xyz21 = tf.concat([tf.expand_dims(wrist_xyz, 0),
                                        keypoint_xyz21[1:, :]], 0)

        data_dict['keypoint_xyz21'] = keypoint_xyz21

        # 2. Read keypoint uv AND VIS
        keypoint_uv_vis21 = tf.reshape(tf.slice(record_bytes_float32, [bytes_read//4], [kp_uv_entries+kp_vis_entries]), [self.num_kp, 3])
        bytes_read += encoding_bytes*(kp_uv_entries+kp_vis_entries)
        keypoint_uv_vis21 = self.convert_kp(keypoint_uv_vis21)
        keypoint_uv21 = keypoint_uv_vis21[:, :2]
        keypoint_vis21 = tf.equal(keypoint_uv_vis21[:, 2], 1.0)

        # calculate wrist vis
        if self.use_wrist_coord:
            wrist_vis = tf.logical_or(keypoint_vis21[16], keypoint_vis21[0])
            keypoint_vis21 = tf.concat([tf.expand_dims(wrist_vis, 0),
                                        keypoint_vis21[1:]], 0)

            wrist_uv = keypoint_uv21[16, :] + 2.0*(keypoint_uv21[0, :] - keypoint_uv21[16, :])
            keypoint_uv21 = tf.concat([tf.expand_dims(wrist_uv, 0),
                                       keypoint_uv21[1:, :]], 0)

        data_dict['keypoint_vis21'] = keypoint_vis21

        if self.coord_uv_noise:
            noise = tf.truncated_normal([42, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv21 += noise

        data_dict['keypoint_uv21'] = keypoint_uv21

        # decode to uint8
        record_bytes_uint8 = tf.decode_raw(value, tf.uint8)

        # 4. Read image
        image = tf.reshape(tf.slice(record_bytes_uint8, [bytes_read], [image_bytes]),
                               [self.image_size[0], self.image_size[1], 3])
        image = tf.cast(image, tf.float32)
        bytes_read += image_bytes

        # subtract mean
        image = image / 255.0 - 0.5
        if self.hue_aug:
            image = tf.image.random_hue(image, self.hue_aug_max)
        data_dict['image'] = image

        """ CONSTANTS """
        # Camera intrinsics
        sx = 822.79041
        sy = 822.79041
        tx = 318.47345
        ty = 250.31296
        data_dict['cam_mat'] = tf.constant([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])

        # Hand side: this dataset only contains left hands
        data_dict['hand_side'] = tf.one_hot(tf.constant(0, dtype=tf.int32), depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

        assert bytes_read == record_bytes, "Doesnt add up."

        """ DEPENDENT DATA ITEMS: XYZ represenations. """
        # make coords relative to root joint
        kp_coord_xyz_root = keypoint_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = keypoint_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        index_root_bone_length = tf.sqrt(tf.reduce_sum(tf.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        data_dict['keypoint_scale'] = index_root_bone_length
        data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / index_root_bone_length  # normalized by length of 12->11

        # calculate local coordinates
        kp_coord_xyz21_local = bone_rel_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_local = tf.squeeze(kp_coord_xyz21_local)
        data_dict['keypoint_xyz21_local'] = kp_coord_xyz21_local

        # calculate viewpoint and coords in canonical coordinates
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_rel_can, rot_mat = tf.squeeze(kp_coord_xyz21_rel_can), tf.squeeze(rot_mat)
        data_dict['keypoint_xyz21_can'] = kp_coord_xyz21_rel_can
        data_dict['rot_mat'] = tf.matrix_inverse(rot_mat)

        """ DEPENDENT DATA ITEMS: HAND CROP """
        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]

            # catch problem, when no valid kp available (happens almost never)
            crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                                  lambda: tf.constant([0.0, 0.0]))
            crop_center.set_shape([2, ])

            if self.crop_center_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_center_noise_sigma)
                crop_center += noise

            crop_scale_noise = tf.constant(1.0)
            if self.crop_scale_noise:
                    crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=1.0, maxval=1.2))

            if not self.use_wrist_coord:
                wrist_uv = keypoint_uv21[16, :] + 2.0*(keypoint_uv21[0, :] - keypoint_uv21[16, :])
                keypoint_uv21 = tf.concat([tf.expand_dims(wrist_uv, 0),
                                           keypoint_uv21[1:, :]], 0)

            # select visible coords only
            kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
            kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
            kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

            # determine size of crop (measure spatial extend of hw coords first)
            min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
            max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), self.image_size)

            # find out larger distance wrt the center of crop
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = tf.reduce_max(crop_size_best)
            crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)

            # catch problem, when no valid kp available
            crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                  lambda: tf.constant(200.0))
            crop_size_best.set_shape([])

            # calculate necessary scaling
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            scale *= crop_scale_noise
            data_dict['crop_scale'] = scale

            if self.crop_offset_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                crop_center += noise

            # Crop image
            img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
            data_dict['image_crop'] = tf.squeeze(img_crop)

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            keypoint_uv21_u = (data_dict['keypoint_uv21'][:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (data_dict['keypoint_uv21'][:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            data_dict['keypoint_uv21'] = keypoint_uv21

            # Modify camera intrinsics
            scale = tf.reshape(scale, [1, ])
            scale_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [scale, [0.0], [0.0],
                                                               [0.0], scale, [0.0],
                                                               [0.0], [0.0], [1.0]])
            scale_matrix = tf.reshape(scale_matrix, [3, 3])

            crop_center_float = tf.cast(crop_center, tf.float32)
            trans1 = crop_center_float[0] * scale - self.crop_size // 2
            trans2 = crop_center_float[1] * scale - self.crop_size // 2
            trans1 = tf.reshape(trans1, [1, ])
            trans2 = tf.reshape(trans2, [1, ])
            trans_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [[1.0], [0.0], -trans2,
                                                               [0.0], [1.0], -trans1,
                                                               [0.0], [0.0], [1.0]])
            trans_matrix = tf.reshape(trans_matrix, [3, 3])

            data_dict['cam_mat'] = tf.matmul(trans_matrix, tf.matmul(scale_matrix, data_dict['cam_mat']))

        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create scoremaps from the subset of 2D annoataion
        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        scoremap_size = self.image_size
        
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                     scoremap_size,
                                                     self.sigma,
                                                     valid_vec=keypoint_vis21)
        
        if self.scoremap_dropout:
            scoremap = tf.nn.dropout(scoremap, self.scoremap_dropout_prob,
                                        noise_shape=[1, 1, 21])
            scoremap *= self.scoremap_dropout_prob

        data_dict['scoremap'] = scoremap

        if self.random_crop_to_size:
            tensor_stack = tf.concat([data_dict['image'],
                                      tf.expand_dims(tf.cast(data_dict['hand_parts'], tf.float32), -1),
                                      tf.cast(data_dict['hand_mask'], tf.float32)], 2)
            s = tensor_stack.get_shape().as_list()
            tensor_stack_cropped = tf.random_crop(tensor_stack,
                                                  [self.random_crop_size, self.random_crop_size, s[2]])
            data_dict = dict()  # delete everything else because the random cropping makes the data invalid anyway
            data_dict['image'], data_dict['hand_parts'], data_dict['hand_mask'] = tensor_stack_cropped[:, :, :3],\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 3], tf.int32),\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 4:], tf.int32)

        names, tensors = zip(*data_dict.items())

        if self.shuffle:
            tensors = tf.train.shuffle_batch_join([tensors],
                                                  batch_size=self.batch_size,
                                                  capacity=100,
                                                  min_after_dequeue=50,
                                                  enqueue_many=False)
        else:
            tensors = tf.train.batch_join([tensors],
                                          batch_size=self.batch_size,
                                          capacity=100,
                                          enqueue_many=False)

        return dict(zip(names, tensors))



    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        with tf.name_scope('create_multiple_gaussian_map'):
            sigma = tf.cast(sigma, tf.float32)
            assert len(output_size) == 2
            s = coords_uv.get_shape().as_list()
            coords_uv = tf.cast(coords_uv, tf.int32)
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
            cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
            cond_in = tf.logical_and(cond_1_in, cond_2_in)
            cond = tf.logical_and(cond_val, cond_in)

            coords_uv = tf.cast(coords_uv, tf.float32)

            # create meshgrid
            x_range = tf.expand_dims(tf.range(output_size[0]), 1)
            y_range = tf.expand_dims(tf.range(output_size[1]), 0)

            X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
            Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

            X.set_shape((output_size[0], output_size[1]))
            Y.set_shape((output_size[0], output_size[1]))

            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)

            X_b = tf.tile(X, [1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, s[0]])

            X_b -= coords_uv[:, 0]
            Y_b -= coords_uv[:, 1]

            dist = tf.square(X_b) + tf.square(Y_b)

            scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

            return scoremap

    @staticmethod
    def convert_kp(keypoints):
        """ Maps the keypoints into the right order. """

        # mapping into my keypoint definition
        kp_dict = {0: 0, 1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 7: 14, 8: 13, 9: 12, 10: 11, 11: 10,
                   12: 9, 13: 8, 14: 7, 15: 6, 16: 5, 17: 4, 18: 3, 19: 2, 20: 1}

        keypoints_new = list()
        for i in range(21):
            if i in kp_dict.keys():
                pos = kp_dict[i]
                keypoints_new.append(keypoints[pos, :])

        return tf.stack(keypoints_new, 0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Test functionality: BASIC
    dataset = BinaryDbReaderSTB(mode='evaluation')
    data = dataset.get()
    session = tf.Session()
    tf.train.start_queue_runners(sess=session)

    for _ in range(10):
        # get data from graph
        image, cam_mat, scoremap, \
        keypoint_uv, keypoint_xyz, keypoint_vis = session.run([data['image'], data['cam_mat'], data['scoremap'],
                                                                      data['keypoint_uv21'], data['keypoint_xyz21'], data['keypoint_vis21']])

        keypoint_vis = np.squeeze(keypoint_vis)
        keypoint_uv = np.squeeze(keypoint_uv)
        keypoint_xyz = np.squeeze(keypoint_xyz)
        cam_mat = np.squeeze(cam_mat)

        # project into frame
        keypoint_uv_proj = np.matmul(keypoint_xyz[:, :], np.transpose(cam_mat[:, :]))
        keypoint_uv_proj = keypoint_uv_proj[:, :2] / keypoint_uv_proj[:, -1:]

        # show results
        fig = plt.figure(1)
        ax1 = fig.add_subplot('221')
        ax2 = fig.add_subplot('222')
        ax3 = fig.add_subplot('223')

        image_rgb = ((np.squeeze(image) + 0.5) * 255.0).astype(np.uint8)
        ax1.imshow(image_rgb)
        ax1.plot(keypoint_uv[keypoint_vis, 0], keypoint_uv[keypoint_vis, 1], 'ro')
        ax2.imshow(image_rgb)
        ax2.plot(keypoint_uv_proj[keypoint_vis, 0], keypoint_uv_proj[keypoint_vis, 1], 'bo')
        scoremap = np.max(scoremap[0, :, :, :] > 0.8, 2)
        ax3.imshow(scoremap)
        plt.show()

    # # Test functionality: CROP
    # dataset = BinaryDbReaderSHB(mode='training', shuffle=False, hand_crop=True, crop_center_noise=True)
    # data = dataset.get()
    # session = tf.Session()
    # tf.train.start_queue_runners(sess=session)
    #
    # for _ in range(5):
    #     # get data from graph
    #     image, \
    #     keypoint_uv21, keypoint_xyz21, keypoint_vis21, \
    #     cam_mat = session.run([data['image_crop'],
    #                              data['keypoint_uv21'], data['keypoint_xyz21'], data['keypoint_vis21'],
    #                              data['cam_mat']])
    #
    #     keypoint_vis21 = np.squeeze(keypoint_vis21)
    #     keypoint_uv21 = np.squeeze(keypoint_uv21)
    #     keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    #     cam_mat = np.squeeze(cam_mat)
    #
    #     # project into frame
    #     keypoint_uv_proj = np.matmul(keypoint_xyz21[:, :], np.transpose(cam_mat[:, :]))
    #     keypoint_uv_proj = keypoint_uv_proj[:, :2] / keypoint_uv_proj[:, -1:]
    #
    #     # show results
    #     fig = plt.figure(1)
    #     ax1 = fig.add_subplot('111')
    #
    #     image_rgb = ((np.squeeze(image) + 0.5) * 255.0).astype(np.uint8)
    #     ax1.imshow(image_rgb)
    #     ax1.plot(keypoint_uv_proj[keypoint_vis21, 0], keypoint_uv_proj[keypoint_vis21, 1], 'ro')
    #     ax1.plot(keypoint_uv21[keypoint_vis21, 0], keypoint_uv21[keypoint_vis21, 1], 'g+')
    #     plt.show()
