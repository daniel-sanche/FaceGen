from NeuralNet import  NeuralNet, NetworkType
from DataLoader import  LoadFilesData, DataLoader
import numpy as np

def trainNetwork(network, saveInterval=500, printInterval=100, goalAcc=0.95, trainDropout=0.5):
    reachedGoal = False
    i = 0
    while not reachedGoal:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batchImage.shape[0], -1])
        if i % printInterval == 0:
            acc = network.printStatus(batchImage, batchSex, batchAge)
            reachedGoal = acc >= goalAcc
        network.train(batchImage, batchSex, batchAge, dropoutVal=trainDropout)
        if (i % saveInterval == 0 and i != 0) or reachedGoal:
            network.saveCheckpoint(saveInterval)
        i = i + 1
    return acc

if __name__ == "__main__":
    # initialize the data loader
    datasetDir = "/home/sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 5
    batch_size = numPerBin * 8 * 2
    noise_size = 100
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size, numWorkerThreads=10, bufferMax=20, debugLogs=False)
    loader.start()

    # start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size,
                              noise_size=noise_size)
    generator = NeuralNet(trainingType=NetworkType.Generator, batch_size=batch_size, image_size=image_size,
            noise_size=noise_size)

    while True:
        print("__DISCRIMINATOR__")
        trainNetwork(discriminator, trainDropout=0.5, goalAcc=0.6)
        print("__GENERATOR__")
        generator.restoreNewestCheckpoint()
        trainNetwork(generator, trainDropout=0.5)
        discriminator.restoreNewestCheckpoint()

