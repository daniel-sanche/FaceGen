from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt
import numpy as np
from nnLayers import create_fully_connected_layer, create_max_pool_layer, create_deconv_layer, create_conv_layer, create_output_layer, create_unpool_layer


num_ages = 100
batch_size = 1000
imageSize = 64
totalPixels = imageSize*imageSize
input_noise_size = 20

fully_connected_size = 64
square_root_fc = int(sqrt(fully_connected_size))

#build the generator network
dropout = tf.placeholder(tf.float32)
gen_input_layer = tf.placeholder(tf.float32, shape=[batch_size, input_noise_size])
gen_fully_connected1, var_dict = create_fully_connected_layer(gen_input_layer, fully_connected_size, input_noise_size,
                                                              dropout, trainable=True, name_prefix="gen_fc")
gen_squared_fc1 = tf.reshape(gen_fully_connected1, [batch_size, square_root_fc, square_root_fc, 1])
#now [1000,8,8,1]
gen_unpool1 = create_unpool_layer(gen_squared_fc1)
#now [1000,16,16,1]
gen_unconv1, var_dict = create_deconv_layer(gen_unpool1, 5, 1, trainable=True, name_prefix="gen_unconv1",
                                            var_dict=var_dict)
#now [1000,16,16,5]
gen_unpool2 = create_unpool_layer(gen_unconv1)
#now [1000,32,32,5]
gen_unconv2, var_dict = create_deconv_layer(gen_unpool2, 5, 5, trainable=True, name_prefix="gen_unconv2",
                                            var_dict=var_dict)
#now [1000,32,32,5]
gen_unpool3 = create_unpool_layer(gen_unconv2)
#now [1000,64,64,5]
gen_unconv3, var_dict = create_deconv_layer(gen_unpool3, 3, 5, trainable=True, name_prefix="gen_unconv3",
                                            var_dict=var_dict)
#now [1000,64,64,5]
gen_output_layer = tf.reshape(gen_unconv3, [batch_size, totalPixels])
#now [1000,64,64,3]

#set up the session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#initialize the data loader
datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
csvPath = "./dataset.csv"
indicesPath = "./indices.p"
csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)
#run in new thread
loader = DataLoader(indices, csvdata, batchSize=batch_size, imageSize=imageSize)
loader.start()


#start training
while True:
    noise_batch = np.random.random_sample((batch_size, input_noise_size))
    feed_dict = {gen_input_layer: noise_batch, dropout: 0.5}
    print("running")
    sess.run(gen_output_layer, feed_dict=feed_dict)
    print("done")
