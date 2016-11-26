from NeuralNet import  NeuralNet
from DataLoader import  LoadFilesData, DataLoader
import numpy as np

def stopFuncGenerator(costDict):
    truth = costDict["truth"]
    return truth > 0.7

def stopFuncDiscriminator(costDict):
    truth =  costDict["truth"]
    sex = costDict["sex"]
    return truth > 0.85 and sex > 0.6


def trainNetwork(network, batchDict, printInterval=100, rounds=100, endEarlyFunc=None):
    i = 0
    endEarly = False
    while i < rounds and not endEarly:
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batchImage.shape[0], -1])
        if i % printInterval == 0:
            network.printStatus(batchImage, batchSex, batchAge)
        network.train(batchImage, batchSex, batchAge)
        i = i + 1
    network.saveCheckpoint(i)

if __name__ == "__main__":
    # initialize the data loader
    datasetDir = "/home/sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 4
    batch_size = numPerBin * 8 * 2
    noise_size = 100
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size, numWorkerThreads=10, bufferMax=20, debugLogs=False)
    loader.start()

    # start training
    network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=noise_size)
    batchDict = loader.getData()

    printInterval = 100
    saveInterval = 1000
    loadedCheckpoint = network.checkpoint_num
    i=0
    while True:
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        if i % printInterval == 0:
            network.printStatus(i+loadedCheckpoint, batchImage, batchSex, batchAge)
        network.train(batchImage, batchSex, batchAge)
        if i % saveInterval == 0 and i != 0:
            network.saveCheckpoint(saveInterval)
        i = i + 1


