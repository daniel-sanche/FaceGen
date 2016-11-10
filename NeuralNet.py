from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt, ceil
import numpy as np
from nnLayers import create_fully_connected_layer, create_max_pool_layer, create_deconv_layer, create_conv_layer, create_output_layer, create_unpool_layer
from Visualization import visualizeImages


class NeuralNet(object):
    def __init__(self, batch_size=1000, image_size=64, noise_size=20):
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_size=noise_size

        self.dropout =  tf.placeholder(tf.float32)
        self._buildGenerator(fcSize=64)
        self._buildDiscriminator(conv1Size=32, conv2Size=64, fcSize=49)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        self.session = sess

    def _buildGenerator(self, fcSize):
        sqrtFc = int(sqrt(fcSize))

        # build the generator network
        gen_input_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_size])
        gen_input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_gender = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_combined = tf.concat(1, [gen_input_age, gen_input_gender, gen_input_noise])
        gen_fully_connected1, var_dict = create_fully_connected_layer(gen_input_combined, fcSize,
                                                                      self.noise_size + 2,
                                                                      self.dropout, trainable=True, name_prefix="gen_fc")
        gen_squared_fc1 = tf.reshape(gen_fully_connected1, [self.batch_size, sqrtFc, sqrtFc, 1])
        # now [1000,8,8,1]
        gen_unpool1 = create_unpool_layer(gen_squared_fc1)
        # now [1000,16,16,1]
        gen_unconv1, var_dict = create_deconv_layer(gen_unpool1, 5, 1, trainable=True, name_prefix="gen_unconv1",
                                                    var_dict=var_dict)
        # now [1000,16,16,5]
        gen_unpool2 = create_unpool_layer(gen_unconv1)
        # now [1000,32,32,5]
        gen_unconv2, var_dict = create_deconv_layer(gen_unpool2, 5, 5, trainable=True, name_prefix="gen_unconv2",
                                                    var_dict=var_dict)
        # now [1000,32,32,5]
        gen_unpool3 = create_unpool_layer(gen_unconv2)
        # now [1000,64,64,5]
        gen_unconv3, var_dict = create_deconv_layer(gen_unpool3, 3, 5, trainable=True, name_prefix="gen_unconv3",
                                                    var_dict=var_dict)
        # now [1000,64,64,5]
        totalPixels = self.image_size * self.image_size * 3
        gen_output_layer = tf.reshape(gen_unconv3, [self.batch_size, totalPixels])
        # now [1000,64,64,3]

        #save important nodes
        self.gen_output = gen_output_layer
        self.gen_input_noise = gen_input_noise
        self.gen_input_age = gen_input_age
        self.gen_input_gender = gen_input_gender
        self.vardict = var_dict

    def _buildDiscriminator(self, conv1Size, conv2Size, fcSize):
        num_pixels = self.image_size * self.image_size * 3
        dis_truth_image = tf.placeholder(tf.float32, shape=[self.batch_size, num_pixels])
        #[1000, 12,288]

        dis_combined_inputs = tf.concat(0, [dis_truth_image, self.gen_output])
        # [2000, 12288]
        dis_reshaped_inputs = tf.reshape(dis_combined_inputs, [self.batch_size * 2, self.image_size, self.image_size, 3])
        # [2000, 64, 64, 3]
        dis_conv1, var_dict = create_conv_layer(dis_reshaped_inputs, conv1Size, 3, trainable=True,
                                                name_prefix="dis_conv1", var_dict=self.vardict)
        # [2000, 64, 64, 32]
        dis_pool1 = create_max_pool_layer(dis_conv1)
        # [2000, 32, 32, 32]
        dis_conv2, var_dict = create_conv_layer(dis_pool1, conv2Size, conv1Size, trainable=True,
                                                name_prefix="dis_conv2", var_dict=var_dict)
        # [2000, 32, 32, 64]
        dis_pool2 = create_max_pool_layer(dis_conv2)
        # [2000, 16, 16, 64]
        dis_pool2_flattened = tf.reshape(dis_pool2, [self.batch_size*2, -1])
        # [2000, 16384]
        dis_fully_connected1, var_dict = create_fully_connected_layer(dis_pool2_flattened, fcSize,
                                                                      16 * 16 * conv2Size, self.dropout,
                                                                      trainable=True,
                                                                      name_prefix="dis_fc", var_dict=var_dict)
        # [2000, 49]
        dis_output_layer, var_dict = create_output_layer(dis_fully_connected1, fcSize, 3,
                                                         trainable=True, name_prefix="dis_out",
                                                         var_dict=var_dict)
        # [2000, 3]
        # save important nodes
        self.dis_truth_image = dis_truth_image
        self.dis_output = dis_output_layer
        self.vardict = var_dict



    def train(self, age_range=[10, 100]):
        batch_size = self.batch_size
        noise_batch = np.random.random_sample((batch_size, self.noise_size))
        ageVec = (
        np.linspace(start=age_range[0], stop=age_range[1], num=batch_size) + np.random.sample(batch_size)).reshape(
            [batch_size, 1])
        genderVec = np.tile(np.array([0, 1], dtype=bool), int(batch_size / 2)).reshape([batch_size, 1])
        feed_dict = {self.gen_input_noise: noise_batch, self.gen_input_age: ageVec, self.gen_input_gender: genderVec, self.dropout: 0.5}
        generatedImages = self.session.run(self.gen_output, feed_dict=feed_dict)
        generatedImages = np.reshape(generatedImages, [batch_size, self.image_size, self.image_size, 3])
        visualizeImages(generatedImages[:50, :, :, :], numRows=5)



#initialize the data loader
datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
csvPath = "./dataset.csv"
indicesPath = "./indices.p"
csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

batch_size = 1000
image_size = 64

#loader = DataLoader(indices, csvdata, batchSize=batch_size, imageSize=image_size)
#loader.start()


#start training
network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=20)
network.train()
print("done")