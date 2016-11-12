from NeuralNet import  NeuralNet, NetworkType
from DataLoader import  LoadFilesData, DataLoader

def trainNetwork(network, lastCost, saveInterval=10, printInterval=10, costReductionGoal=0.9):
    network.restoreNewestCheckpoint()
    reachedGoal = False
    i = 0
    while not reachedGoal:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batchImage.shape[0], -1])
        cost = network.train(batchImage, batchSex, batchAge, print_results=(i % printInterval == 0))
        if i % saveInterval == 0 and i != 0:
            discriminator.saveCheckpoint(saveInterval)
        i = i + 1
        reachedGoal = cost <= lastCost * costReductionGoal
    #rached goal. Save state final time
    discriminator.saveCheckpoint(i%saveInterval)
    return cost

if __name__ == "__main__":
    # initialize the data loader
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

    # start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size,
                              noise_size=20)
    generator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size,
                              noise_size=20)

    disCurrentCost = float("inf")
    genCurrentCost = float("inf")

    while True:
        print("__GENERATOR__")
        genCurrentCost = trainNetwork(generator, genCurrentCost)
        print("__DISCRIMINATOR__")
        disCurrentCost = trainNetwork(discriminator, disCurrentCost)

