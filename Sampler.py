from NeuralNet import  NeuralNet
from DataLoader import  LoadFilesData, DataLoader
from Visualization import visualizeImages
from math import ceil, sqrt
import numpy as np


def randomSample(network, sampleSize, gender=None, age=None, saveName=None):
    if gender is not None:
        genderVec = np.ones([sampleSize, 1]) * (gender != 0)
    else:
        genderVec = np.random.randint(2, size=sampleSize)
    if age is not None:
        ageVec = np.ones([sampleSize, 1]) * age
    else:
        ageVec = np.random.randint(15, 75, size=sampleSize)
    genderVec = ((genderVec * 2) - 1).astype(np.float32).reshape([-1, 1])
    ageVec = (((ageVec / 100.0) * 2) - 1).astype(np.float32).reshape([-1, 1])
    noiseVec = np.random.uniform(-1, 1, [sampleSize, network.noise_size]).astype(np.float32)
    samples =  network.getSample(noiseVec, genderVec, ageVec)
    if saveName is not None:
        numRows = int(ceil(sqrt(sampleSize)))
        visualizeImages(samples, numRows=numRows, fileName=saveName)
    return samples

def ageSample(network, numImages, minAge=25, maxAge=75, gender=None, noiseArr=None, saveName=None):
    if gender is None:
        gender = np.random.randint(2, size=1)
    if noiseArr is None:
        noiseArr = np.random.uniform(-1, 1, [1, network.noise_size]).astype(np.float32)
    ageMat = (((np.linspace(minAge, maxAge, numImages, dtype=int) / 100.0) * 2) - 1).reshape([numImages, 1])
    genderMat = ((np.ones([numImages, 1]) * gender) * 2) - 1
    noiseMat = np.ones([numImages, network.noise_size]) * noiseArr
    samples = network.getSample(noiseMat, genderMat, ageMat)
    if saveName is not None:
        visualizeImages(samples, numRows=1, fileName=saveName)
    return samples


if __name__ == "__main__":
    # initialize the data loader
    image_size = 64
    batch_size = 64
    noise_size = 100

    # start training
    network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=noise_size, learningRate=5e-4)

    randomSample(network, 36, saveName="sample.png")
    ageSample(network, 10, saveName="age_sample.png")
