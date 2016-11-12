from NeuralNet import  NeuralNet, NetworkType
from DataLoader import  LoadFilesData, DataLoader


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
    i = 0
    while True:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batch_size, -1])
        discriminator.train(batchImage, batchSex, batchAge, print_results=i % saveSteps == 0)
        if i % saveSteps == 0 and i != 0:
            discriminator.saveCheckpoint(saveSteps)
        i = i + 1