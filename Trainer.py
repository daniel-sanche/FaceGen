from NeuralNet import  NeuralNet
from DataLoader import  LoadFilesData, DataLoader
import numpy as np

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
    network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=noise_size, learningRate=5e-4)

    printInterval = 100
    saveInterval = 1000
    loadedCheckpoint = network.checkpoint_num
    i=0
    while True:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        if i % printInterval == 0:
            network.printStatus(i+loadedCheckpoint, batchImage, batchSex, batchAge)
        network.train(batchImage, batchSex, batchAge)
        if i % saveInterval == 0 and i != 0:
            network.saveCheckpoint(saveInterval)
        i = i + 1


