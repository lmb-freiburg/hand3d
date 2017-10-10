import tensorflow as tf


def _stitch_mat_from_vecs(vector_list):
    """ Stitches a given list of vectors into a 4x4 matrix.

        Input:
            vector_list: list of 16 tensors, which will be stitched into a matrix. list contains matrix elements
                in a row-first fashion (m11, m12, m13, m14, m21, m22, m23, m24, ...). Length of the vectors has
                to be the same, because it is interpreted as batch dimension.
    """

    assert len(vector_list) == 16, "There have to be exactly 16 tensors in vector_list."
    batch_size = vector_list[0].get_shape().as_list()[0]
    vector_list = [tf.reshape(x, [1, batch_size]) for x in vector_list]

    trafo_matrix = tf.dynamic_stitch([[0], [1], [2], [3],
                                      [4], [5], [6], [7],
                                      [8], [9], [10], [11],
                                      [12], [13], [14], [15]], vector_list)

    trafo_matrix = tf.reshape(trafo_matrix, [4, 4, batch_size])
    trafo_matrix = tf.transpose(trafo_matrix, [2, 0, 1])

    return trafo_matrix


def _atan2(y, x):
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


def _get_rot_mat_x_hom(angle):
    """ Returns a 3D rotation matrix in homogeneous coords.  """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec, zero_vec,
                                          zero_vec, tf.cos(angle), -tf.sin(angle), zero_vec,
                                          zero_vec, tf.sin(angle), tf.cos(angle), zero_vec,
                                          zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _get_rot_mat_y_hom(angle):
    """ Returns a 3D rotation matrix in homogeneous coords.  """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([tf.cos(angle), zero_vec, tf.sin(angle), zero_vec,
                                          zero_vec, one_vec, zero_vec, zero_vec,
                                          -tf.sin(angle), zero_vec, tf.cos(angle), zero_vec,
                                          zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _get_rot_mat_z_hom(angle):
    """ Returns a 3D rotation matrix in homogeneous coords. """
    one_vec = tf.ones_like(angle)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([tf.cos(angle), -tf.sin(angle), zero_vec, zero_vec,
                                          tf.sin(angle), tf.cos(angle), zero_vec, zero_vec,
                                          zero_vec, zero_vec, one_vec, zero_vec,
                                          zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _get_trans_mat_hom(trans):
    """ Returns a 3D translation matrix in homogeneous coords. """
    one_vec = tf.ones_like(trans)
    zero_vec = one_vec*0.0
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec, zero_vec,
                                          zero_vec, one_vec, zero_vec, zero_vec,
                                          zero_vec, zero_vec, one_vec, trans,
                                          zero_vec, zero_vec, zero_vec, one_vec])
    return trafo_matrix


def _to_hom(vector):
    s = vector.get_shape().as_list()
    vector = tf.reshape(vector, [s[0], -1, 1])
    vector = tf.concat([vector, tf.ones((s[0], 1, 1))], 1)
    return vector


def _from_hom(vector):
    s = vector.get_shape().as_list()
    vector = tf.reshape(vector, [s[0], -1, 1])
    return vector[:, :-1, :]


def _forward(length, angle_x, angle_y, T):
    """ Given a articulations it calculates the update to the coord matrix and the location of the end point in global coords. """
    # update current transformation from local -> new local
    T_this = tf.matmul(_get_trans_mat_hom(-length), tf.matmul(_get_rot_mat_x_hom(-angle_x), _get_rot_mat_y_hom(-angle_y)))

    # trafo from global -> new local
    T = tf.matmul(T_this, T)

    # calculate global location of this point
    # x0 = tf.constant([[0.0], [0.0], [0.0], [1.0]])
    s = length.get_shape().as_list()
    x0 = _to_hom(tf.zeros((s[0], 3, 1)))
    x = tf.matmul(tf.matrix_inverse(T), x0)
    return x, T


def _backward(delta_vec, T):
    """ Given a vector it calculates the articulated angles and updates the current coord matrix. """
    # calculate length directly
    length = tf.sqrt(delta_vec[:, 0, 0]**2 + delta_vec[:, 1, 0]**2 + delta_vec[:, 2, 0]**2)

    # calculate y rotation
    angle_y = _atan2(delta_vec[:, 0, 0], delta_vec[:, 2, 0])

    # this vector is an intermediate result and always has x=0
    delta_vec_tmp = tf.matmul(_get_rot_mat_y_hom(-angle_y), delta_vec)

    # calculate x rotation
    angle_x = _atan2(-delta_vec_tmp[:, 1, 0], delta_vec_tmp[:, 2, 0])

    # update current transformation from local -> new local
    T_this = tf.matmul(_get_trans_mat_hom(-length), tf.matmul(_get_rot_mat_x_hom(-angle_x), _get_rot_mat_y_hom(-angle_y)))

    # trafo from global -> new local
    T = tf.matmul(T_this, T)

    # make them all batched scalars
    length = tf.reshape(length, [-1])
    angle_x = tf.reshape(angle_x, [-1])
    angle_y = tf.reshape(angle_y, [-1])
    return length, angle_x, angle_y, T

# Encodes how the kinematic chain goes; Is a mapping from child -> parent: dict[child] = parent
kinematic_chain_dict = {0: 'root',

                        4: 'root',
                        3: 4,
                        2: 3,
                        1: 2,

                        8: 'root',
                        7: 8,
                        6: 7,
                        5: 6,

                        12: 'root',
                        11: 12,
                        10: 11,
                        9: 10,

                        16: 'root',
                        15: 16,
                        14: 15,
                        13: 14,

                        20: 'root',
                        19: 20,
                        18: 19,
                        17: 18}

# order in which we will calculate stuff
kinematic_chain_list = [0,
                        4, 3, 2, 1,
                        8, 7, 6, 5,
                        12, 11, 10, 9,
                        16, 15, 14, 13,
                        20, 19, 18, 17]


def bone_rel_trafo(coords_xyz):
    """ Transforms the given real xyz coordinates into a bunch of relative frames.
        The frames are set up according to the kinematic chain. Each parent of the chain
        is the origin for the location of the next bone, where the z-axis is aligned with the bone
        and articulation is measured as rotations along the x- and y- axes.

        Inputs:
            coords_xyz: BxNx3 matrix, containing the coordinates for each of the N keypoints
    """
    with tf.variable_scope('bone_rel_transformation'):
        coords_xyz = tf.reshape(coords_xyz, [-1, 21, 3])

        # list of results
        trafo_list = [None for _ in kinematic_chain_list]
        coords_rel_list = [0.0 for _ in kinematic_chain_list]

        # Iterate kinematic chain list (from root --> leaves)
        for bone_id in kinematic_chain_list:

            # get parent of current bone
            parent_id = kinematic_chain_dict[bone_id]

            if parent_id == 'root':

                # if there is no parent global = local
                delta_vec = _to_hom(tf.expand_dims(coords_xyz[:, bone_id, :], 1))
                T = _get_trans_mat_hom(tf.zeros_like(coords_xyz[:, 0, 0]))

                # get articulation angles from bone vector
                results = _backward(delta_vec, T)

                # save results
                coords_rel_list[bone_id] = tf.stack(results[:3], 1)
                trafo_list[bone_id] = results[3]

            else:
                T = trafo_list[parent_id]  #by sticking to the order defined in kinematic_chain_list its ensured, that this is avail
                assert T is not None, 'Something went wrong.'

                # calculate coords in local system
                x_local_parent = tf.matmul(T, _to_hom(tf.expand_dims(coords_xyz[:, parent_id, :], 1)))
                x_local_child = tf.matmul(T, _to_hom(tf.expand_dims(coords_xyz[:, bone_id, :], 1)))

                # calculate bone vector in local coords
                delta_vec = x_local_child - x_local_parent
                delta_vec = _to_hom(tf.expand_dims(delta_vec[:, :3, :], 1))

                # get articulation angles from bone vector
                results = _backward(delta_vec, T)

                # save results
                coords_rel_list[bone_id] = tf.stack(results[:3], 1)
                trafo_list[bone_id] = results[3]

        coords_rel = tf.stack(coords_rel_list, 1)

        return coords_rel


def bone_rel_trafo_inv(coords_rel):
    """ Assembles relative coords back to xyz coords. Inverse operation to bone_rel_trafo().

        Inputs:
            coords_rel: BxNx3 matrix, containing the coordinates for each of the N keypoints [length, angle_x, angle_y]
    """
    with tf.variable_scope('assemble_bone_rel'):
        s = coords_rel.get_shape().as_list()
        if len(s) == 2:
            coords_rel = tf.expand_dims(coords_rel, 0)
            s = coords_rel.get_shape().as_list()
        assert len(s) == 3, "Has to be a batch of coords."

        # list of results
        trafo_list = [None for _ in kinematic_chain_list]
        coords_xyz_list = [0.0 for _ in kinematic_chain_list]

        # Iterate kinematic chain list (from root --> leaves)
        for bone_id in kinematic_chain_list:

            # get parent of current bone
            parent_id = kinematic_chain_dict[bone_id]

            if parent_id == 'root':
                # if there is no parent global = local
                T = _get_trans_mat_hom(tf.zeros_like(coords_rel[:, 0, 0]))

                # get articulation angles from bone vector
                x, T = _forward(length=coords_rel[:, bone_id, 0],
                                angle_x=coords_rel[:, bone_id, 1],
                                angle_y=coords_rel[:, bone_id, 2],
                                T=T)

                # save results
                coords_xyz_list[bone_id] = tf.squeeze(_from_hom(x), [2])
                trafo_list[bone_id] = T

            else:
                T = trafo_list[parent_id]  #by sticking to the order defined in kinematic_chain_list its ensured, that this is avail
                assert T is not None, 'Something went wrong.'

                # get articulation angles from bone vector
                x, T = _forward(length=coords_rel[:, bone_id, 0],
                                angle_x=coords_rel[:, bone_id, 1],
                                angle_y=coords_rel[:, bone_id, 2],
                                T=T)

                # save results
                coords_xyz_list[bone_id] = tf.squeeze(_from_hom(x), [2])
                trafo_list[bone_id] = T

        coords_xyz = tf.stack(coords_xyz_list, 1)
        return coords_xyz