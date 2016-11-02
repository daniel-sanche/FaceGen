import sys
import os
import pandas as pd
import  numpy as np
import pickle
from DataLoader import getBatch
from math import ceil
from scipy.misc import  imsave, imresize

def visualizeBatch(batchOutput, indices,fileName="batch.png", maxImgSize=64):
    imageVec = batchOutput["image"]
    sexVec = batchOutput["sex"]
    ageVec = batchOutput["age"]

    imageSize = [imageVec.shape[1], imageVec.shape[2], imageVec.shape[3]]
    numItems = sexVec.shape[0]
    numRows = len(indices["AgeBinLimits"]) * 2
    numCols = int(ceil(numItems/numRows))

    CombinedImage = np.ones([imageSize[0]*numRows, imageSize[1]*numCols, imageSize[2]])
    i = 0
    rowStart = 0
    for r in range(numRows):
        rowEnd = rowStart + imageSize[0]
        RowImage = np.zeros([imageSize[0], imageSize[1]*numCols, imageSize[2]])
        lastStart = 0
        for c in range(numCols):
            thisImage = imageVec[i,:,:]
            #thisSex = sexVec[i]
            #thisAge = ageVec[i]
            end = lastStart + imageSize[1]
            RowImage[:, lastStart:end, :] = thisImage
            lastStart = end
            i = i + 1
            if i >= numItems:
                break
        CombinedImage[rowStart:rowEnd,:,:] = RowImage
        rowStart = rowEnd
    if maxImgSize is not None:
        CombinedImage = imresize(CombinedImage, [maxImgSize*numRows, maxImgSize*numCols])
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
    visualizeBatch(batchData, indices)