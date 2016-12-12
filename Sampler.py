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

def ageSample(network, numAges, minAge=25, maxAge=75, gender=None, noiseArr=None, saveName=None):
    if gender is None:
        gender = np.random.randint(2, size=1)
    if noiseArr is None:
        noiseArr = np.random.uniform(-1, 1, [1, network.noise_size]).astype(np.float32)
    ageMat = (((np.linspace(minAge, maxAge, numAges, dtype=int) / 100.0) * 2) - 1).reshape([numAges, 1])
    genderMat = ((np.ones([numAges, 1]) * gender) * 2) - 1
    noiseMat = np.ones([numAges, network.noise_size]) * noiseArr
    samples = network.getSample(noiseMat, genderMat, ageMat)
    if saveName is not None:
        visualizeImages(samples, numRows=1, fileName=saveName)
    return samples

def ageSampleMultiple(network, numAges, numSamples, minAge=25, maxAge=75, saveName=None):
    combinedMat = np.zeros([numSamples*numAges, 64, 64, 3])
    for i in range(numSamples):
        result = ageSample(network, numAges, minAge=minAge, maxAge=maxAge, saveName=None)
        combinedMat[numAges*i:numAges*(i+1),:,:,:] = result
    if saveName is not None:
        visualizeImages(combinedMat, numRows=numSamples, fileName=saveName)
    return combinedMat

def sexSample(network, numSamples, age=None, saveName=None):
    if age is not None:
        ageVec = np.ones([numSamples, 1]) * age
    else:
        ageVec = np.random.randint(15, 75, size=numSamples)
    ageVec = (((ageVec / 100.0) * 2) - 1).astype(np.float32).reshape([-1, 1])
    noiseArr = np.random.uniform(-1, 1, [numSamples, network.noise_size]).astype(np.float32)
    genderArr = np.array([0,1])


    noiseArr = np.concatenate([noiseArr, noiseArr])
    ageVec = np.concatenate([ageVec, ageVec])
    genderArr = genderArr.repeat(numSamples).reshape(numSamples*2, 1)

    samples = network.getSample(noiseArr, genderArr, ageVec)
    if saveName is not None:
        visualizeImages(samples, numRows=2, fileName=saveName)
    return samples

if __name__ == "__main__":
    # initialize the data loader
    image_size = 64
    batch_size = 64
    noise_size = 100

    # start training
    network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=noise_size, learningRate=5e-4)

    randomSample(network, 36, saveName="sample.png")
    ageSampleMultiple(network, 10, 4, saveName="age_sample.png")
    sexSample(network, 10, saveName="sex_sample.png")
