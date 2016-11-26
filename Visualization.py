import sys
import os
import pandas as pd
import  numpy as np
import pickle
from DataLoader import getBatch
from math import ceil
from scipy.misc import  imsave, imresize

"""
function to visualize a batch of data from the dataset
Will display the entire batch, with each row being a separate age group/gender (from young to old)

Params
    batchOutput:    the dictionary output from getBatch
    indices:        the indices file for the dataset
    fileName:       the name of the output png
    maxImgSize:     the size of all sub-images that are combined into the final output
"""
def visualizeBatch(batchOutput, indices,fileName="batch.png", maxImgSize=64):
    imageVec = batchOutput["image"]
    numItems = imageVec.shape[0]
    numRows = len(indices["AgeBinLimits"]) * 2
    numCols = int(ceil(numItems/numRows))
    visualizeImages(imageVec, numRows=numRows, numCols=numCols, maxImgSize=maxImgSize, fileName=fileName)

"""
More general visualization function, that can be used for any set of images (not just from dataset)
takes in a numpy array of images ([batchSize, rows, cols, channels]), and displays
a subset of the images in a png file

Params
    imageMat:   a numpy array of images to display
    numRows:    the number of rows to use in the output
    maxImgSize:     the size of all sub-images that are combined into the final output
    fileName:       the name of the output png
"""
def visualizeImages(imageMat, numRows=5, maxImgSize=64, fileName="images_set.png"):
    #create directory if necessary
    path = os.path.dirname(os.path.abspath(fileName))
    if not os.path.exists(path):
        os.mkdir(path)

    numItems = imageMat.shape[0]
    numCols = int(ceil(numItems / numRows))
    imageSize = [imageMat.shape[1], imageMat.shape[2], imageMat.shape[3]]
    CombinedImage = np.ones([imageSize[0] * numRows, imageSize[1] * numCols, imageSize[2]])
    i = 0
    rowStart = 0
    for r in range(numRows):
        rowEnd = rowStart + imageSize[0]
        RowImage = np.zeros([imageSize[0], imageSize[1] * numCols, imageSize[2]])
        lastStart = 0
        for c in range(numCols):
            thisImage = imageMat[i, :, :, :]
            end = lastStart + imageSize[1]
            RowImage[:, lastStart:end, :] = thisImage
            lastStart = end
            i = i + 1
            if i >= numItems:
                break
        CombinedImage[rowStart:rowEnd, :, :] = RowImage
        rowStart = rowEnd
    if maxImgSize is not None:
        CombinedImage = imresize(CombinedImage, [maxImgSize * numRows, maxImgSize * numCols])
    if os.path.exists(fileName):
        os.remove(fileName)
    imsave(fileName, CombinedImage)

def csvFromOutput(isTruth, age, gender,guessedTrue, guessedAge, guessedGender, fileName="batchResults.csv"):
    combined = np.concatenate((isTruth, age, gender, guessedTrue, guessedAge, guessedGender), axis=1)
    titles = ["isReal","Age","Sex","GuesedReal","GuessedAge","GuessedSex"]
    df = pd.DataFrame(combined, columns=titles)
    df.to_csv(fileName)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("requires 2 parameters (csv_path, indices_path)")
        exit()

    csvPath = sys.argv[1]
    indicesPath = sys.argv[2]

    if not os.path.exists(csvPath) or not os.path.exists(indicesPath):
        print("one or both files not found")
        exit()

    print("restoring csv data...")
    csvdata = pd.read_csv(csvPath)

    print("restoring indices data...")
    file = open(indicesPath, "rb")
    indices = pickle.load(file)
    file.close()

    state = None
    batchData, state, didFinish = getBatch(indices, csvdata, prevState=state, imageSize=64)
    visualizeBatch(batchData, indices)
