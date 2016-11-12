from NeuralNet import  NeuralNet, NetworkType
from DataLoader import  LoadFilesData, DataLoader
import numpy as np

def trainNetwork(network, lastCost, saveInterval=500, printInterval=100, costReductionGoal=0.9, trainDropout=0.5):
    reachedGoal = False
    i = 0
    cost = lastCost
    goalCost = lastCost * costReductionGoal
    print ("Goal Cost: " + str(goalCost))
    while not reachedGoal:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batchImage.shape[0], -1])
        if i % printInterval == 0:
            cost = network.printStatus(batchImage, batchSex, batchAge)
            reachedGoal = cost <= goalCost
        network.train(batchImage, batchSex, batchAge, dropoutVal=trainDropout)
        if (i % saveInterval == 0 and i != 0) or reachedGoal:
            network.saveCheckpoint(saveInterval)
        i = i + 1
    return cost

if __name__ == "__main__":
    # initialize the data loader
    datasetDir = "/home/sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 15
    batch_size = numPerBin * 8 * 2
    noise_size = image_size * image_size * 3
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size, numWorkerThreads=10, bufferMax=20, debugLogs=False)
    loader.start()

    # start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size,
                              noise_size=noise_size)
    generator = NeuralNet(trainingType=NetworkType.Generator, batch_size=batch_size, image_size=image_size,
            noise_size=noise_size)

    disCurrentCost = float("inf")
    genCurrentCost = float("inf")

    while True:
        print("__GENERATOR__")
        genCurrentCost = trainNetwork(generator, genCurrentCost, trainDropout=0.5)
        print("__DISCRIMINATOR__")
        discriminator.restoreNewestCheckpoint()
        disCurrentCost = trainNetwork(discriminator, disCurrentCost, trainDropout=0.7)
        generator.restoreNewestCheckpoint()

