from NeuralNet import  NeuralNet
from DataLoader import  LoadFilesData, DataLoader
from Visualization import visualizeImages

if __name__ == "__main__":
    # initialize the data loader
    saveSteps = 10
    image_size = 64
    numPerBin = 4
    batch_size = numPerBin * 8 * 2
    noise_size = 500

    # start training
    network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=noise_size, learningRate=5e-4)

    sample = network.randomSample(100)
    visualizeImages(sample, numRows=10, fileName="sample.png")