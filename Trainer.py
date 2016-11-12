from NeuralNet import  NeuralNet, NetworkType
from DataLoader import  LoadFilesData, DataLoader
import numpy as np

def trainNetwork(network, lastCost, saveInterval=10, printInterval=2, costReductionGoal=0.9):
    reachedGoal = False
    i = 0
    while not reachedGoal:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batchImage.shape[0], -1])
        if i % printInterval == 0:
            network.printStatus(batchImage, batchSex, batchAge)
        cost = network.train(batchImage, batchSex, batchAge)
        if i % saveInterval == 0 and i != 0:
            discriminator.saveCheckpoint(saveInterval)
        i = i + 1
        reachedGoal = cost <= lastCost * costReductionGoal
    #reached goal. Save state final time
    discriminator.saveCheckpoint(i%saveInterval)
    return cost

if __name__ == "__main__":
    # initialize the data loader
    datasetDir = "/home/sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 10
    batch_size = numPerBin * 8 * 2
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size)
    loader.start()

    # start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size,
                              noise_size=20)
    generator = NeuralNet(trainingType=NetworkType.Generator, batch_size=batch_size, image_size=image_size,
                              noise_size=20)

    disCurrentCost = float("inf")
    genCurrentCost = float("inf")

    while True:
        print("__GENERATOR__")
        genCurrentCost = trainNetwork(generator, genCurrentCost)
        print("__DISCRIMINATOR__")
        discriminator.restoreNewestCheckpoint()
        disCurrentCost = trainNetwork(discriminator, disCurrentCost)
        generator.restoreNewestCheckpoint()

