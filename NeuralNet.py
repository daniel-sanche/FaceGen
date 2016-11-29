from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt
import numpy as np
from Visualization import visualizeImages, csvFromOutput
from enum import Enum
from os import walk, path, mkdir
import pandas as pd
import pickle

class NeuralNet(object):
    """"""
    """
    Neural Net Layers
    """
    def create_conv_layer(self, prev_layer, new_depth, prev_depth, name_prefix="conv", patch_size=3):
        W, b = self.create_variables([patch_size, patch_size, prev_depth, new_depth], [new_depth],
                                          name_prefix=name_prefix)
        new_layer = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(new_layer + b)

    def create_max_pool_layer(self, prev_layer):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def create_fully_connected_layer(self, prev_layer, new_size, prev_size, name_prefix="fc"):
        W, b = self.create_variables([prev_size, new_size], [new_size], name_prefix=name_prefix)
        new_layer = tf.nn.relu(tf.matmul(prev_layer, W) + b)
        return new_layer

    def create_output_layer(self, prev_layer, prev_size, num_classes, name_prefix="out"):
        W, b = self.create_variables([prev_size, num_classes], [num_classes], name_prefix=name_prefix)
        return tf.matmul(prev_layer, W) + b

    def create_deconv_layer(self, prev_layer, new_depth, prev_depth, name_prefix="deconv", patch_size=3, relu=True):
        input_shape = prev_layer.get_shape().as_list()
        new_shape = input_shape
        new_shape[-1] = new_depth
        W, b = self.create_variables([patch_size, patch_size, new_depth, prev_depth], [new_depth],
                                          name_prefix=name_prefix)
        new_layer = tf.nn.conv2d_transpose(prev_layer, W, new_shape, strides=[1, 1, 1, 1], padding='SAME')
        if relu:
            return tf.nn.relu(new_layer + b)
        else:
            return new_layer + b

    def create_variables(self, w_size, b_size, name_prefix="untitled", w_stddev=0.02, b_val=0.1):
        W_name = name_prefix + "-W"
        b_name = name_prefix + "-b"
        W = tf.Variable(tf.truncated_normal(w_size, stddev=w_stddev), name=W_name)
        b = tf.Variable(tf.constant(b_val, shape=b_size), name=b_name)
        return W, b

    def create_batchnorm_layer(self, prev_layer, layer_shape, name_prefix="bnorm"):
        scale_name = name_prefix + "-S"
        offset_name = name_prefix + "-O"
        mean, variance = tf.nn.moments(prev_layer, axes=[0])
        scale = tf.Variable(tf.ones(layer_shape), name=scale_name)
        offset = tf.Variable(tf.zeros(layer_shape), name=offset_name)
        return tf.nn.batch_normalization(prev_layer, mean, variance, offset, scale, 1e-8)

    def create_upsample_layer(self, prev_layer, new_size):
        resized = tf.image.resize_images(prev_layer, new_size, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return resized

    """
    Initialization Helpers
    """
    def __init__(self, batch_size=1000, chkptDir="./checkpoints", chkptName="FaceGen.ckpt",image_size=64, noise_size=1000, age_range=[10, 100], learningRate=1e-4):
        self.age_range = age_range
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_size=noise_size

        self._buildGenerator(firstImageSize=8)
        self._buildDiscriminator(conv1Size=64, conv2Size=32, fcSize=49)
        self._buildCostFunctions(learningRate=learningRate)

        #create a constant noise vector for printing, so we can watch images improve over time
        print_noise_path = "print_noise.p"
        if path.exists(print_noise_path):
            file = open(print_noise_path, "rb")
            self.print_noise = pickle.load(file)
        else:
            #use the same noise values for men and women to see how similar they are
            noise_single = np.random.uniform(-1, 1, [self.batch_size/ 2, self.noise_size])
            self.print_noise = np.concatenate([noise_single, noise_single]).reshape([self.batch_size,self.noise_size])
            file = open(print_noise_path, "wb")
            pickle.dump(self.print_noise, file)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        self.session = sess
        self.saver = tf.train.Saver(max_to_keep=3)
        self.checkpoint_name = chkptName
        self.checkpoint_dir = chkptDir
        self.checkpoint_num = 0
        self.restoreNewestCheckpoint()
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        pd.set_option('expand_frame_repr', False)

    def _buildGenerator(self, firstImageSize=8):

        # build the generator network
        self.input_sex = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        self.input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        self.input_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_size])
        combined_inputs = tf.concat(1, [self.input_sex, self.input_age, self.input_noise])
        gen_fully_connected1 = self.create_fully_connected_layer(combined_inputs, firstImageSize * firstImageSize * 64,
                                                                 self.noise_size+2,
                                                                 name_prefix="gen_fc")
        gen_squared_fc1 = tf.reshape(gen_fully_connected1, [self.batch_size, firstImageSize, firstImageSize, 64])
        gen_squared_fc1_norm = self.create_batchnorm_layer(gen_squared_fc1, [8,8,64], name_prefix="gen_fc")
        # s[1000,8,8,64]
        gen_unpool1 = self.create_upsample_layer(gen_squared_fc1_norm, 16)
        # [1000,16,16,64]
        gen_unconv1 = self.create_deconv_layer(gen_unpool1, 32, 64, name_prefix="gen_unconv1")
        gen_unconv1_norm = self.create_batchnorm_layer(gen_unconv1, [16, 16, 32],name_prefix="gen_unconv1")
        # [1000,16,16,32]
        gen_unpool2 = self.create_upsample_layer(gen_unconv1_norm, 32)
        # [1000,32,32,32]
        gen_unconv2 = self.create_deconv_layer(gen_unpool2, 16, 32, name_prefix="gen_unconv2")
        gen_unconv2_norm = self.create_batchnorm_layer(gen_unconv2, [32,32,16],name_prefix="gen_unconv2")
        # [1000,32,32,16]
        gen_unpool3 = self.create_upsample_layer(gen_unconv2_norm, 64)
        # [1000,64,64,16]
        gen_unconv3 = self.create_deconv_layer(gen_unpool3, 3, 16, name_prefix="gen_unconv3")
        gen_unconv3_norm = self.create_batchnorm_layer(gen_unconv3, [64,64,3],name_prefix="gen_unconv3")
        # [1000,64,64,3]
        self.gen_output = tf.nn.tanh(gen_unconv3_norm)
        # [1000,12288]

    def _buildDiscriminator(self, conv1Size, conv2Size, fcSize):
        self.dis_input_image = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
        dis_combined_inputs = tf.concat(0, [self.gen_output, self.dis_input_image])
        #[2000, 64, 64, 3]

        #combine sex and age as new channels on the image
        sex_channel = tf.ones([self.batch_size, self.image_size*self.image_size]) * self.input_sex
        sex_channel = tf.concat(0, [sex_channel, sex_channel])
        sex_channel = tf.reshape(sex_channel, [self.batch_size*2, 64, 64, 1])
        age_channel = tf.ones([self.batch_size, self.image_size * self.image_size]) * self.input_age
        age_channel =  tf.concat(0, [age_channel, age_channel])
        age_channel = tf.reshape(age_channel, [self.batch_size * 2, 64, 64, 1])
        combined_channels = tf.concat(3, [dis_combined_inputs, sex_channel, age_channel])

        # [2000, 64, 64, 3]
        dis_conv1 = self.create_conv_layer(combined_channels, conv1Size, 5, name_prefix="dis_conv1")
        # [2000, 64, 64, 64]
        dis_pool1 = self.create_max_pool_layer(dis_conv1)
        # [2000, 32, 32, 64]
        dis_conv2 = self.create_conv_layer(dis_pool1, conv2Size, conv1Size, name_prefix="dis_conv2")
        # [2000, 32, 32, 32]
        dis_pool2 = self.create_max_pool_layer(dis_conv2)
        # [2000, 16, 16, 32]
        dis_pool2_flattened = tf.reshape(dis_pool2, [self.batch_size*2, -1])
        # [2000, 8192]
        dis_fully_connected1 = self.create_fully_connected_layer(dis_pool2_flattened, fcSize,
                                                                      16 * 16 * conv2Size,
                                                                      name_prefix="dis_fc")
        # [2000, 49]
        self.dis_output = self.create_output_layer(dis_fully_connected1,fcSize,1,name_prefix="dis_out")
        # [2000, 3]


    def _buildCostFunctions(self, learningRate=2e-4, beta1=0.5):
        generated_logits, true_logits = tf.split(0, 2, self.dis_output);

        self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(true_logits, tf.ones([self.batch_size, 1])))
        self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_logits, tf.zeros([self.batch_size, 1])))
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_logits, tf.ones([self.batch_size, 1])))

        t_vars = tf.trainable_variables()
        dis_vars = [var for var in t_vars if 'dis_' in var.name]
        gen_vars = [var for var in t_vars if 'gen_' in var.name]

        self.dis_train = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(self.dis_loss, var_list=dis_vars)
        self.gen_train = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(self.gen_loss, var_list=gen_vars)


    """
    Public Functions (Interface)
    """

    def saveCheckpoint(self, runsSinceLast):
        if runsSinceLast > 0:
            self.checkpoint_num = self.checkpoint_num + runsSinceLast
            self.saver.save(self.session, self.checkpoint_dir + "/" + self.checkpoint_name, self.checkpoint_num)
            print(self.checkpoint_name + " " + str(self.checkpoint_num) + " saved")


    def restoreNewestCheckpoint(self):
        if not path.exists(self.checkpoint_dir):
            mkdir(self.checkpoint_dir)
        highest_found = 0
        path_found = None
        for subdir, dirs, files in walk(self.checkpoint_dir):
            for file in files:
                if self.checkpoint_name in file and ".meta" not in file and ".txt" not in file:
                    iteration_num = int(file.split("-")[-1])
                    if iteration_num >= highest_found:
                        highest_found = iteration_num
                        path_found = path.join(subdir, file)
        if path_found is not None:
            #if existing one was found, restore previous checkpoint
            print ("restoring checkpoint ", path_found)
            self.saver.restore(self.session, path_found)
        else:
            print("no checkpoint found named " + self.checkpoint_name + " in " + self.checkpoint_dir)
        self.checkpoint_num = highest_found

    def train(self, truthImages, truthGenders, truthAges):
        noise_batch = np.random.uniform(-1, 1, [self.batch_size, self.noise_size]).astype(np.float32)
        feed_dict = {self.input_noise: noise_batch, self.input_age: truthAges, self.input_sex: truthGenders,
                     self.dis_input_image: truthImages}
        errFake, errReal, gen_cost = self.session.run((self.dis_loss_fake, self.dis_loss_real, self.gen_loss), feed_dict=feed_dict)
        dis_cost = errFake + errReal
        if gen_cost/dis_cost < 3:
            self.session.run((self.dis_train), feed_dict=feed_dict)
        if dis_cost/gen_cost < 3:
            self.session.run((self.gen_train), feed_dict=feed_dict)

    def printStatus(self,num, truthImages, truthGenders, truthAges):
        feed_dict = {self.input_noise: self.print_noise, self.input_age: truthAges, self.input_sex: truthGenders,
                     self.dis_input_image: truthImages}

        runList = (self.dis_loss_fake, self.dis_loss_real, self.gen_loss)
        errFake, errReal, errGen = self.session.run(runList, feed_dict=feed_dict)
        print("round: "  + str(num) + " d_loss: " + str(errFake+errReal) + ", g_loss: " + str(errGen))

        #render images to files
        printSexLabels = np.repeat([-1,1],self.batch_size/2).reshape([self.batch_size, 1])
        ageRange = np.linspace(-0.7, 0.7, self.batch_size/2)
        printAgeLabels = np.concatenate([ageRange, ageRange]).reshape([self.batch_size, 1])

        feed_dict = {self.input_noise: self.print_noise, self.input_age: printAgeLabels, self.input_sex: printSexLabels,
                     self.dis_input_image: truthImages}
        outImages = self.session.run(self.gen_output, feed_dict=feed_dict)
        outImages = (outImages + 1.0) / 2.0
        visualizeImages(outImages, numRows=8, fileName="./images/run_" + str(num) + ".png" )
        visualizeImages(outImages, numRows=8, fileName="output.png" )
        truthImages = (truthImages + 1.0) / 2.0
        visualizeImages(truthImages, numRows=8, fileName="last_batch.png")

