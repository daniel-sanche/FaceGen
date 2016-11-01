import sys
import os
import pandas as pd
import  numpy as np
import pickle
from DataLoader import getBatch
from math import ceil, sqrt
from scipy.misc import  imsave, imresize

def visualizeBatch(batchOutput, fileName="batch.png", maxSize=[5000, 5000, 3]):
    imageVec = batchOutput["image"]
    sexVec = batchOutput["sex"]
    ageVec = batchOutput["age"]

    imageSize = [imageVec.shape[1], imageVec.shape[2], imageVec.shape[3]]
    numItems = sexVec.shape[0]
    sqrtItems = sqrt(numItems)
    numRows = int(ceil(sqrtItems))
    numCols = numRows

    CombinedImage = np.ones([imageSize[0]*numRows, imageSize[1]*numCols, imageSize[2]])
    i = 0
    rowStart = 0
    for r in range(numRows):
        rowEnd = rowStart + imageSize[0]
        RowImage = np.zeros([imageSize[0], imageSize[1]*numCols, imageSize[2]])
        lastStart = 0
        for c in range(numCols):
            thisImage = imageVec[i,:,:]
            thisSex = sexVec[i]
            thisAge = ageVec[i]
            end = lastStart + imageSize[1]
            RowImage[:, lastStart:end, :] = thisImage
            lastStart = end
            i = i + 1
            if i >= numItems:
                break
        CombinedImage[rowStart:rowEnd,:,:] = RowImage
        rowStart = rowEnd
    if maxSize is not None:
        CombinedImage = imresize(CombinedImage, maxSize)
    imsave(fileName, CombinedImage)



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
    batchData, state, didFinish = getBatch(indices, csvdata, prevState=state)
    visualizeBatch(batchData)