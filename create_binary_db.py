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
"""
    Script to convert Rendered Handpose Dataset into binary files,
    which allows for much faster reading than plain image files.

    Set "path_to_db" and "set" accordingly.

    In order to use this file you need to download and unzip the dataset first.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import scipy.misc
import struct

# SET THIS to where RHD is located on your machine
path_to_db = './RHD_published_v2/'

# chose if you want to create a binary for training or evaluation set
# set = 'training'
set = 'evaluation'

### No more changes below this line ###


# function to write the binary file
def write_to_binary(file_handle, image, mask, kp_coord_xyz, kp_coord_uv, kp_visible, K_mat):
    """" Writes records to an open binary file. """
    bytes_written = 0
    # 1. write kp_coord_xyz
    for coord in kp_coord_xyz:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
        file_handle.write(struct.pack('f', coord[2]))
    bytes_written += 4*kp_coord_xyz.shape[0]*kp_coord_xyz.shape[1]

    # 2. write kp_coord_uv
    for coord in kp_coord_uv:
        file_handle.write(struct.pack('f', coord[0]))
        file_handle.write(struct.pack('f', coord[1]))
    bytes_written += 4*kp_coord_uv.shape[0]*kp_coord_uv.shape[1]

    # 3. write camera intrinsic matrix
    for K_row in K_mat:
        for K_element in K_row:
            file_handle.write(struct.pack('f', K_element))
    bytes_written += 4*9

    file_handle.write(struct.pack('B', 255))
    file_handle.write(struct.pack('B', 255))
    bytes_written += 2

    # 4. write image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            file_handle.write(struct.pack('B', image[x, y, 0]))
            file_handle.write(struct.pack('B', image[x, y, 1]))
            file_handle.write(struct.pack('B', image[x, y, 2]))
    bytes_written += 4*image.shape[0]*image.shape[1]*image.shape[2]

    # 5. write mask
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            file_handle.write(struct.pack('B', mask[x, y]))
    bytes_written += 4*mask.shape[0]*mask.shape[1]

    # 6. write visibility
    for x in range(kp_visible.shape[0]):
        file_handle.write(struct.pack('B', kp_visible[x]))
    bytes_written += kp_visible.shape[0]

    # print('bytes_written', bytes_written)

# binary file we will write
file_name_out = './data/bin/rhd_%s.bin' % set

if not os.path.exists('./data/bin'):
    os.mkdir('./data/bin')

# load annotations of this set
with open(os.path.join(path_to_db, set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# iterate samples of the set and write to binary file
with open(file_name_out, 'wb') as fo:
    num_samples = len(anno_all.items())
    for sample_id, anno in anno_all.items():
        # load data
        image = scipy.misc.imread(os.path.join(path_to_db, set, 'color', '%.5d.png' % sample_id))
        mask = scipy.misc.imread(os.path.join(path_to_db, set, 'mask', '%.5d.png' % sample_id))

        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
        kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

        write_to_binary(fo, image, mask, kp_coord_xyz, kp_coord_uv, kp_visible, camera_intrinsic_matrix)

        if (sample_id % 100) == 0:
            print('%d / %d images done: %.3f percent' % (sample_id, num_samples, sample_id*100.0/num_samples))
