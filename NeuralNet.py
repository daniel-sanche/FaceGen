from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt
import numpy as np
from Visualization import visualizeImages, csvFromOutput
from enum import Enum
from os import walk, path, mkdir
import pandas as pd

class NetworkType(Enum):
    Generator=0,
    Discriminator=1

class NeuralNet(object):
    """"""
    """
    Neural Net Layers
    """
    def create_conv_layer(self, prev_layer, new_depth, prev_depth, trainable=True, name_prefix="conv", patch_size=3):
        W, b = self.create_variables([patch_size, patch_size, prev_depth, new_depth], [new_depth],
                                          name_prefix=name_prefix, trainable=trainable)
        new_layer = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(new_layer + b)

    def create_max_pool_layer(self, prev_layer):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def create_fully_connected_layer(self,prev_layer, new_size, prev_size, dropout_prob, trainable=True, name_prefix="fc"):
        W, b = self.create_variables([prev_size, new_size], [new_size], name_prefix=name_prefix,trainable=trainable)
        new_layer = tf.nn.relu(tf.matmul(prev_layer, W) + b)
        return tf.nn.dropout(new_layer, dropout_prob)

    def create_output_layer(self, prev_layer, prev_size, num_classes, trainable=True, name_prefix="out"):
        W, b = self.create_variables([prev_size, num_classes], [num_classes], name_prefix=name_prefix,trainable=trainable)
        return tf.nn.sigmoid(tf.matmul(prev_layer, W) + b)

    def create_deconv_layer(self, prev_layer, new_depth, prev_depth, trainable=True, name_prefix="deconv", patch_size=3):
        input_shape = prev_layer.get_shape().as_list()
        new_shape = input_shape
        new_shape[-1] = new_depth
        W, b = self.create_variables([patch_size, patch_size, new_depth, prev_depth], [new_depth],
                                          name_prefix=name_prefix, trainable=trainable)
        new_layer = tf.nn.conv2d_transpose(prev_layer, W, new_shape, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(new_layer + b)

    def create_variables(self, w_size, b_size, name_prefix="untitled", trainable=True, w_stddev=0.1, b_val=0.1):
        W_name = name_prefix + "-W"
        b_name = name_prefix + "-b"
        W = tf.Variable(tf.truncated_normal(w_size, stddev=w_stddev),
                        trainable=trainable, name=W_name)
        b = tf.Variable(tf.constant(b_val, shape=b_size), trainable=trainable, name=b_name)
        self.vardict[W_name] = W
        self.vardict[b_name] = b
        return W, b

    def create_unpool_layer(self, prev_layer):
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

    """
    Initialization Helpers
    """
    def __init__(self, trainingType, batch_size=1000, chkptDir="./checkpoints", chkptName="FaceGen.ckpt",image_size=64, noise_size=20, age_range=[10, 100], learningRate=1e-4):
        self.vardict = {}
        self.trainingType = trainingType
        self.age_range = age_range
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_size=noise_size

        self.dropout =  tf.placeholder(tf.float32)
        trainGen = (trainingType==NetworkType.Generator)
        self._buildGenerator(fcSize=64, train=trainGen)
        self._buildDiscriminator(conv1Size=32, conv2Size=64, fcSize=49, train=(not trainGen))
        self._buildCostFunctions(learningRate=learningRate)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        self.session = sess
        self.saver = tf.train.Saver(self.vardict, max_to_keep=3)
        self.checkpoint_name = chkptName
        self.checkpoint_dir = chkptDir
        self.checkpoint_num = 0
        self.restoreNewestCheckpoint()
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        pd.set_option('expand_frame_repr', False)

    def _buildGenerator(self, fcSize, train=True):
        sqrtFc = int(sqrt(fcSize))

        # build the generator network
        gen_input_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_size])
        gen_input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_gender = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_combined = tf.concat(1, [gen_input_age, gen_input_gender, gen_input_noise])
        gen_fully_connected1 = self.create_fully_connected_layer(gen_input_combined, fcSize,
                                                                 self.noise_size + 2,
                                                                 self.dropout, trainable=train, name_prefix="gen_fc")
        gen_squared_fc1 = tf.reshape(gen_fully_connected1, [self.batch_size, sqrtFc, sqrtFc, 1])
        # now [1000,8,8,1]
        gen_unpool1 = self.create_unpool_layer(gen_squared_fc1)
        # now [1000,16,16,1]
        gen_unconv1 = self.create_deconv_layer(gen_unpool1, 5, 1, trainable=train, name_prefix="gen_unconv1")
        # now [1000,16,16,5]
        gen_unpool2 = self.create_unpool_layer(gen_unconv1)
        # now [1000,32,32,5]
        gen_unconv2 = self.create_deconv_layer(gen_unpool2, 5, 5, trainable=train, name_prefix="gen_unconv2")
        # now [1000,32,32,5]
        gen_unpool3 = self.create_unpool_layer(gen_unconv2)
        # now [1000,64,64,5]
        gen_unconv3 = self.create_deconv_layer(gen_unpool3, 3, 5, trainable=train, name_prefix="gen_unconv3")
        # now [1000,64,64,5]
        totalPixels = self.image_size * self.image_size * 3
        gen_output_layer = tf.reshape(gen_unconv3, [self.batch_size, totalPixels])
        # now [1000,64,64,3]

        #save important nodes
        self.gen_output = gen_output_layer
        self.gen_input_noise = gen_input_noise
        self.gen_input_age = gen_input_age
        self.gen_input_gender = gen_input_gender

    def _buildDiscriminator(self, conv1Size, conv2Size, fcSize, train=True):
        num_pixels = self.image_size * self.image_size * 3
        dis_input_image = tf.placeholder(tf.float32, shape=[self.batch_size, num_pixels])
        dis_input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        dis_input_gender = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        dis_labels_truth = tf.concat(0, [tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1])])
        dis_labels_age = tf.concat(0, [dis_input_age, self.gen_input_age])
        dis_labels_gender = tf.concat(0, [dis_input_gender, self.gen_input_gender])
        #[1000, 12,288]

        dis_combined_inputs = tf.concat(0, [dis_input_image, self.gen_output])
        # [2000, 12288]
        dis_reshaped_inputs = tf.reshape(dis_combined_inputs, [self.batch_size * 2, self.image_size, self.image_size, 3])
        # [2000, 64, 64, 3]
        dis_conv1 = self.create_conv_layer(dis_reshaped_inputs, conv1Size, 3, trainable=train, name_prefix="dis_conv1")
        # [2000, 64, 64, 32]
        dis_pool1 = self.create_max_pool_layer(dis_conv1)
        # [2000, 32, 32, 32]
        dis_conv2 = self.create_conv_layer(dis_pool1, conv2Size, conv1Size, trainable=train, name_prefix="dis_conv2")
        # [2000, 32, 32, 64]
        dis_pool2 = self.create_max_pool_layer(dis_conv2)
        # [2000, 16, 16, 64]
        dis_pool2_flattened = tf.reshape(dis_pool2, [self.batch_size*2, -1])
        # [2000, 16384]
        dis_fully_connected1 = self.create_fully_connected_layer(dis_pool2_flattened, fcSize,
                                                                      16 * 16 * conv2Size, self.dropout,
                                                                      trainable=train,
                                                                      name_prefix="dis_fc")
        # [2000, 49]
        dis_output_layer = self.create_output_layer(dis_fully_connected1,fcSize,3,trainable=train,name_prefix="dis_out")
        # [2000, 3]

        # save important nodes
        self.dis_input_image = dis_input_image
        self.dis_input_age = dis_input_age
        self.dis_input_gender = dis_input_gender
        self.dis_output = dis_output_layer
        self.dis_label_truth = dis_labels_truth
        self.dis_label_age = dis_labels_age
        self.dis_label_gender = dis_labels_gender

    def _buildCostFunctions(self, learningRate=1e-4, ageDiffForAcc=0.1):
        truthOutput, ageOutput, sexOutput = tf.split(1, 3, self.dis_output)
        scaledAge = tf.scalar_mul(self.age_range[1], ageOutput)
        truthDiff = tf.sub(truthOutput, self.dis_label_truth)
        ageDiff = tf.sub(scaledAge, self.dis_label_age) / self.age_range[1]
        sexDiff = tf.sub(sexOutput, self.dis_label_gender)

        #discriminator cost for true images is the sum of the error in the truth, gender, and age values
        #error in generated images is only the error in the truth value, because we don't want to learn false patterns
        disTruthCost = tf.nn.l2_loss(truthDiff) / self.batch_size * 2
        disSexCost = tf.nn.l2_loss(tf.mul(sexDiff, self.dis_label_truth)) / self.batch_size
        disAgeCost = tf.nn.l2_loss(tf.mul(ageDiff, self.dis_label_truth)) / self.batch_size
        disCombinedCost = tf.add(tf.add(disSexCost, disAgeCost), disTruthCost)
        disTrainStep = tf.train.AdamOptimizer(learningRate).minimize(disCombinedCost)

        #generator cost is the L2 loss of truth, gender, and age values
        #truth value is weighted, because making realistic humans should be a higher priority
        fakeLabels = tf.cast(tf.logical_not(tf.cast(self.dis_label_truth, tf.bool)), tf.float32)
        genTruthDiff = tf.mul(tf.sub(truthOutput, fakeLabels), fakeLabels)
        genAgeDiff = tf.mul(ageDiff, fakeLabels)
        genSexDiff = tf.mul(sexDiff, fakeLabels)
        genTruthCost = tf.nn.l2_loss(genTruthDiff) / self.batch_size
        genAgeCost = tf.nn.l2_loss(genAgeDiff) / self.batch_size
        genSexCost = tf.nn.l2_loss(genSexDiff) / self.batch_size
        genCombinedCost = tf.add(tf.add(genTruthCost, genAgeCost), genSexCost)
        genTrainStep = tf.train.AdamOptimizer(learningRate).minimize(genCombinedCost)

        #calculate the accuracy for all predictions
        sexCorrect = tf.cast(tf.equal(tf.round(self.dis_label_gender), tf.round(sexOutput)), tf.float32)
        ageCorrect = tf.cast(tf.less(tf.abs(ageDiff), ageDiffForAcc), tf.float32)

        disTruthCorrect = tf.equal(tf.round(self.dis_label_truth), tf.round(truthOutput))
        disTruthAccuracy = tf.reduce_mean(tf.cast(disTruthCorrect, tf.float32))
        disSexCorrect = tf.mul(sexCorrect, self.dis_label_truth)
        disSexAccuracy =  tf.scalar_mul(2, tf.reduce_mean(tf.cast(disSexCorrect, tf.float32)))
        disAgeCorrect = tf.mul(ageCorrect, self.dis_label_truth)
        disAgeAccuracy = tf.scalar_mul(2, tf.reduce_mean(tf.cast(disAgeCorrect, tf.float32)))

        fooledTruthArr = tf.equal(tf.mul(tf.round(truthOutput), fakeLabels), tf.ones_like(truthOutput))
        genTruthAccuracy = tf.scalar_mul(2, tf.reduce_mean(tf.cast(fooledTruthArr, tf.float32)))
        genAgeCorrect =  tf.mul(ageCorrect, fakeLabels)
        genAgeAccuracy = tf.scalar_mul(2, tf.reduce_mean(tf.cast(genAgeCorrect, tf.float32)))
        genSexCorrect = tf.mul(sexCorrect, fakeLabels)
        genSexAccuracy = tf.scalar_mul(2, tf.reduce_mean(tf.cast(genSexCorrect, tf.float32)))

        self.dis_out_age = scaledAge
        self.dis_out_gender = sexOutput
        self.dis_out_truth = truthOutput
        self.dis_train = disTrainStep
        self.dis_cost_total = disCombinedCost
        self.dis_cost_truth = disTruthCost
        self.dis_cost_age = disAgeCost
        self.dis_cost_sex = disSexCost
        self.dis_accuracy_truth = disTruthAccuracy
        self.dis_accuracy_age = disAgeAccuracy
        self.dis_accuracy_sex = disSexAccuracy
        self.gen_train = genTrainStep
        self.gen_cost_total = genCombinedCost
        self.gen_cost_age = genAgeCost
        self.gen_cost_sex = genSexCost
        self.gen_cost_truth = genTruthCost
        self.gen_accuracy_truth = genTruthAccuracy
        self.gen_accuracy_age = genAgeAccuracy
        self.gen_accuracy_sex = genSexAccuracy

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


    def _createFeedDict(self, truthImages, truthGenders, truthAges, dropout=0.5):
        batch_size = self.batch_size
        noise_batch = np.random.random_sample((batch_size, self.noise_size))
        ageVec = (
        np.linspace(start=self.age_range[0], stop=self.age_range[1], num=batch_size) + np.random.sample(batch_size))
        ageVec = ageVec.reshape([batch_size, 1])
        genderVec = np.tile(np.array([0, 1], dtype=bool), int(batch_size / 2)).reshape([batch_size, 1])
        feed_dict = {self.gen_input_noise: noise_batch, self.gen_input_age: ageVec,
                     self.gen_input_gender: genderVec, self.dropout: dropout, self.dis_input_gender: truthGenders,
                     self.dis_input_age: truthAges, self.dis_input_image: truthImages}
        return feed_dict, ageVec, genderVec

    def train(self, truthImages, truthGenders, truthAges):
        feed_dict,_,_ = self._createFeedDict(truthImages, truthGenders, truthAges, dropout=0.5)
        if self.trainingType == NetworkType.Discriminator:
            _, cost =self.session.run((self.dis_train, self.dis_cost_total), feed_dict=feed_dict)
        else:
            _, cost = self.session.run((self.gen_train, self.gen_cost_total), feed_dict=feed_dict)
        return cost

    def printStatus(self, truthImages, truthGenders, truthAges):
        feed_dict, ageVec, genderVec = self._createFeedDict(truthImages, truthGenders, truthAges, dropout=1)

        if self.trainingType == NetworkType.Discriminator:
            outputList = (self.dis_out_truth, self.dis_out_age, self.dis_out_gender,
                          self.dis_cost_total, self.dis_cost_truth, self.dis_cost_age, self.dis_cost_sex,
                          self.dis_accuracy_truth, self.dis_accuracy_age,self.dis_accuracy_sex)
            outT, outA, outS, costTot, costT, costA, costS, accT,accA, accS = self.session.run(outputList, feed_dict=feed_dict)
            df = pd.DataFrame(np.array([costTot,costT, costA, costS, accT, accA, accS]).reshape(1,-1), columns=["Total Cost","Truth Cost","Age Cost","Sex Cost","Truth Acc","Age Acc","Sex Acc"], index=["Discriminator"])
            print(df)
        else:
            outputList = (self.gen_output, self.dis_out_truth, self.dis_out_age, self.dis_out_gender,
                          self.gen_cost_total,self.gen_cost_truth, self.gen_cost_age, self.gen_cost_sex,
                          self.gen_accuracy_truth,self.gen_accuracy_age,self.gen_accuracy_sex)
            outImages, outT, outA, outS, costTot, costT, costA, costS, accT,accA, accS = self.session.run(outputList, feed_dict=feed_dict)
            df = pd.DataFrame(np.array([costTot, costT, costA, costS, accT, accA, accS]).reshape(1, -1),
                              columns=["Total Cost", "Truth Cost", "Age Cost", "Sex Cost", "Truth Acc", "Age Acc",
                                       "Sex Acc"], index=["Generator"])
            print(df)
            outImages = np.reshape(outImages, [self.batch_size, self.image_size, self.image_size, 3])
            visualizeImages(outImages[:50, :, :, :], numRows=5)
        csvFromOutput(np.concatenate([np.ones([self.batch_size, 1]), np.zeros([self.batch_size, 1])]),
                      np.concatenate([truthAges, ageVec]),
                      np.concatenate([truthGenders, genderVec]),
                      outT, outA, outS)
        return costTot


if __name__ == "__main__":
    #initialize the data loader
    datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 10
    batch_size = numPerBin * 8 * 2
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size)
    loader.start()

    #start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size, noise_size=20)
    i=0
    while True:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batch_size, -1])
        discriminator.train(batchImage, batchSex, batchAge, print_results=i%saveSteps==0)
        if i%saveSteps==0 and i != 0:
            discriminator.saveCheckpoint(saveSteps)
        i=i+1