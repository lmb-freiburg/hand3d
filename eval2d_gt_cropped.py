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
""" Script for isolated evaluation of PoseNet on hand cropped images.
    Ground truth keypoint annotations are used for crop generation.

    This allows to reproduce row 1 from Table 1 of the paper:
    GT R-val AUC=0.724 EPE median=5.001 EPE mean=9.135
"""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, EvalUtil, load_weights_from_snapshot

# flag that allows to load a retrained snapshot(original weights used in the paper are used otherwise)
USE_RETRAINED = False
PATH_TO_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true

# get dataset
dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)

# build network graph
data = dataset.get()

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = ColorHandPose3DNetwork()
keypoints_scoremap = net.inference_pose2d(data['image_crop'])
keypoints_scoremap = keypoints_scoremap[-1]

# upscale to original size
s = data['image_crop'].get_shape().as_list()
keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network weights
if USE_RETRAINED:
    # retrained version
    last_cpt = tf.train.latest_checkpoint(PATH_TO_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
else:
    # load weights used in the paper
    net.init(sess, weight_files=['./weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    crop_scale, keypoints_scoremap_v, kp_uv21_gt, kp_vis = sess.run([data['crop_scale'], keypoints_scoremap, data['keypoint_uv21'], data['keypoint_vis21']])

    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    kp_uv21_gt = np.squeeze(kp_uv21_gt)
    kp_vis = np.squeeze(kp_vis)
    crop_scale = np.squeeze(crop_scale)

    # detect keypoints
    coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1)

    util.feed(kp_uv21_gt/crop_scale, kp_vis, coord_uv_pred_crop/crop_scale)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

mean, median, auc, _, _ = util.get_measures(0.0, 30.0, 20)
print('Evaluation results:')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)
