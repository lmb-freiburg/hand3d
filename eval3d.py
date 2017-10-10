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
""" Script for evaluation of different Lifting variants on full scale images.

    This allows to reproduce Table 2 of the paper R-val "Average median error":
    Method      | Number in the paper                   | Our result with TF 1.3

    Direct      | 20.9                                  | 20.848 mm
    Bottleneck  | Number in the paper is *not* correct  | 21.907 mm
    Local       | 39.1                                  | 39.121 mm
    Proposed    | 18.8                                  | 18.840 mm


    Also there is one new variant that was not included in the paper as it is more current work.
    Its the like local, but with the loss in xyz coordinate frame, which seems to work better:
    Local with XYZ Loss  21.950 mm
"""""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from nets.PosePriorNetwork import PosePriorNetwork
from utils.general import EvalUtil, load_weights_from_snapshot

# Chose which variant to evaluate
USE_RETRAINED = False
VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
# VARIANT = 'proposed'

# get dataset
dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)

# build network graph
data = dataset.get()

# build network
net = PosePriorNetwork(VARIANT)

# feed through network
evaluation = tf.placeholder_with_default(True, shape=())
coord3d_pred, _, _ = net.inference(data['scoremap'], data['hand_side'], evaluation)

coord3d_gt = data['keypoint_xyz21']

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network with weights used in the paper
if USE_RETRAINED:
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_%s_drop/' % VARIANT)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
else:
    net.init(sess, weight_files=['./weights/lifting-%s.pickle' % VARIANT])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoint_xyz21, keypoint_scale, coord3d_pred_v = sess.run([data['keypoint_xyz21'], data['keypoint_scale'], coord3d_pred, ])

    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    keypoint_scale = np.squeeze(keypoint_scale)
    coord3d_pred_v = np.squeeze(coord3d_pred_v)

    # rescale to meters
    coord3d_pred_v *= keypoint_scale

    # center gt
    keypoint_xyz21 -= keypoint_xyz21[0, :]

    kp_vis = np.ones_like(keypoint_xyz21[:, 0])
    util.feed(keypoint_xyz21, kp_vis, coord3d_pred_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, _, _ = util.get_measures(0.0, 0.050, 20)
print('Evaluation results for %s:' % VARIANT)
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))
print('Area under curve: %.3f' % auc)
