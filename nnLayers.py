import tensorflow as tf
import numpy as np


def create_conv_layer(prev_layer, new_depth, prev_depth, trainable=True, name_prefix="conv", var_dict={}, patch_size=3):
    W, b, var_dict = create_variables([patch_size, patch_size, prev_depth, new_depth], [new_depth],
                                      name_prefix=name_prefix, var_dict=var_dict, trainable=trainable)
    new_layer = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(new_layer + b), var_dict


def create_max_pool_layer(prev_layer):
    return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_fully_connected_layer(prev_layer, new_size, prev_size, dropout_prob, trainable=True, name_prefix="fc",
                                 var_dict={}):
    W, b, var_dict = create_variables([prev_size, new_size], [new_size], name_prefix=name_prefix, trainable=trainable,
                                      var_dict=var_dict)
    new_layer = tf.nn.relu(tf.matmul(prev_layer, W) + b)
    return tf.nn.dropout(new_layer, dropout_prob), var_dict


def create_output_layer(prev_layer, prev_size, num_classes, trainable=True, name_prefix="out", var_dict={}):
    W, b, var_dict = create_variables([prev_size, num_classes], [num_classes], name_prefix=name_prefix,
                                      trainable=trainable, var_dict=var_dict)
    return tf.nn.sigmoid(tf.matmul(prev_layer, W) + b), var_dict


def create_deconv_layer(prev_layer, new_depth, prev_depth, trainable=True, name_prefix="deconv", var_dict={},
                        patch_size=3):
    input_shape = prev_layer.get_shape().as_list()
    new_shape = input_shape
    new_shape[-1] = new_depth
    W, b, var_dict = create_variables([patch_size, patch_size, new_depth, prev_depth], [new_depth],
                                      name_prefix=name_prefix, trainable=trainable, var_dict=var_dict)
    new_layer = tf.nn.conv2d_transpose(prev_layer, W, new_shape, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(new_layer + b), var_dict


def create_variables(w_size, b_size, name_prefix="untitled", trainable=True, var_dict={}, w_stddev=0.1, b_val=0.1):
    W_name = name_prefix + "-W"
    b_name = name_prefix + "-b"
    W = tf.Variable(tf.truncated_normal(w_size, stddev=w_stddev),
                    trainable=trainable, name=W_name)
    b = tf.Variable(tf.constant(b_val, shape=b_size), trainable=trainable, name=b_name)
    var_dict[W_name] = W
    var_dict[b_name] = b
    return W, b, var_dict


def create_unpool_layer(prev_layer):
    shape_list = prev_layer.get_shape().as_list()
    channel_size = shape_list[-1]
    x_size = shape_list[-3]
    y_size = shape_list[-2]
    batch_size = shape_list[0]
    batch_layers = channel_size * batch_size
    spacers = tf.zeros([1, (y_size * x_size * 3) * batch_layers])
    val_indices = np.array([x for x in range(0, x_size * y_size * 4 * batch_layers) if
                            x % 2 == 0 and int(x // (x_size * 2)) % 2 == 0]).reshape(shape_list)
    space_indices = np.array([x for x in range(0, x_size * y_size * 4 * batch_layers) if
                              x % 2 != 0 or int(x // (x_size * 2)) % 2 != 0]).reshape([1, -1])
    stitched = tf.dynamic_stitch([val_indices, space_indices], [prev_layer, spacers])
    new_size = shape_list
    new_size[-3] = x_size * 2
    new_size[-2] = y_size * 2
    reshaped_stitched = tf.reshape(stitched, new_size)
    return reshaped_stitched
