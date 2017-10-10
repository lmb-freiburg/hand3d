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
import sys

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from data.BinaryDbReader import BinaryDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot

# training parameters
train_para = {'lr': [1e-5, 1e-6, 1e-7],
              'lr_iter': [20000, 30000],
              'max_iter': 40000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_handsegnet'}

# get dataset
dataset = BinaryDbReader(mode='training',
                         batch_size=8, shuffle=True,
                         hue_aug=True, random_crop_to_size=True)

# build network graph
data = dataset.get()

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = ColorHandPose3DNetwork()
hand_mask_pred = net.inference_detection(data['image'], train=True)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
s = data['hand_mask'].get_shape().as_list()
for i, pred_item in enumerate(hand_mask_pred):
    gt = tf.reshape(data['hand_mask'], [s[0]*s[1]*s[2], -1])
    pred = tf.reshape(hand_mask_pred, [s[0]*s[1]*s[2], -1])
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=gt))

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

rename_dict = {'CPM/PersonNet': 'HandSegNet',
               '_CPM': ''}
load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PoseNet', 'Mconv', 'conv6'], rename_dict)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
for i in range(train_para['max_iter']):
    _, loss_v = sess.run([train_op, loss])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
