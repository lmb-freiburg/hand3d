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
""" Script for evaluation of HandSegNet + PoseNet on full scale images.

    To reproduce row 3 from Table 1 of the paper set USE_RETRAINED = False, use_wrist_coord=True, scale_to_size=True):
    Net R-val AUC=0.663 EPE median=5.833 EPE mean=17.041

    Using the correct evaluation setting (use_wrist_coord=False) leads to:
    Net R-val AUC=0.679 EPE median=5.275 EPE mean=16.561

    Using the correct evaluation setting and reporting results in the 320x320 frame of RHD
    (use_wrist_coord=False, scale_to_size=True) frame leads to:
    Net R-val AUC=0.635 EPE median=6.745 EPE mean=18.741
"""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, EvalUtil, load_weights_from_snapshot

# flag that allows to load a retrained snapshot(original weights used in the paper are used otherwise)
USE_RETRAINED = False
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'  # only used when USE_RETRAINED is true

# get dataset
dataset = BinaryDbReader(mode='evaluation', shuffle=False, use_wrist_coord=True, scale_to_size=True)

# build network graph
data = dataset.get()

# build network
net = ColorHandPose3DNetwork()

# scale input to common size for evaluation
image_scaled = tf.image.resize_images(data['image'], (240, 320))
s = data['image'].get_shape().as_list()
scale = (240.0/s[1], 320.0/s[2])

# feed trough network
keypoints_scoremap, _, scale_crop, center = net.inference2d(image_scaled)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network weights
if USE_RETRAINED:
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    # retrained version: PoseNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
else:
    # load weights used in the paper
    net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle',
                                 './weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoints_scoremap_v,\
    scale_crop_v, center_v, kp_uv21_gt, kp_vis = sess.run([keypoints_scoremap, scale_crop, center, data['keypoint_uv21'], data['keypoint_vis21']])

    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    kp_uv21_gt = np.squeeze(kp_uv21_gt)
    kp_vis = np.squeeze(kp_vis)

    # detect keypoints
    coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw_pred = trafo_coords(coord_hw_pred_crop, center_v, scale_crop_v, 256)
    coord_uv_pred = np.stack([coord_hw_pred[:, 1], coord_hw_pred[:, 0]], 1)

    # scale pred to image size of the dataset (to match with stored coordinates)
    coord_uv_pred[:, 1] /= scale[0]
    coord_uv_pred[:, 0] /= scale[1]

    # some datasets are already stored with downsampled resolution
    scale2orig_res = 1.0
    if hasattr(dataset, 'resolution'):
        scale2orig_res = dataset.resolution

    util.feed(kp_uv21_gt/scale2orig_res, kp_vis, coord_uv_pred/scale2orig_res)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, _, _ = util.get_measures(0.0, 30.0, 20)
print('Evaluation results:')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)