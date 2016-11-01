import os
import numpy as np
import pandas as pd
import pickle
from DataLoader import createCsv, createIndices
from PIL import Image
"""
creates a csv file detailing the age/gender breakdown of the csv dataset

Params
    csvdata:    a pandas dataframe from the csv file we are using as out dataset
    ageRange:   the start and end values for the ages we are using
    outPath:    the path to save the stats csv at

Returns
    0: a pandas dataframe representing the results
"""
def statsCsv(csvdata, ageRange=[15, 100], outPath="stats.csv"):
    numRows = len(csvdata.index)
    numAges = ageRange[1] - ageRange[0] + 1
    resultsArr = np.zeros([numAges, 2], dtype=int)
    for i in range(numRows):
        sex = int(csvdata["isMale"][i])
        age = int(csvdata["age"][i])
        adjustedAge = age - ageRange[0]
        resultsArr[adjustedAge, sex] = resultsArr[adjustedAge, sex] + 1
    df = pd.DataFrame(resultsArr, columns=["female", "male"], index=np.arange(ageRange[0], ageRange[1]+1))
    df.to_csv(outPath)
    return df

def findImageSizeRange(csvData):
    numRows = len(csvData.index)
    minRes = float("inf")
    maxRes = 0
    minResStr = ""
    maxResStr = ""
    sumRes = 0
    for i in range(numRows):
        width = csvData["image_width"][i]
        height = csvdata["image_height"][i]
        resStr = "[" + str(width) + ", " + str(height) + "]"
        res = height * width
        sumRes = sumRes + res

        if res > maxRes:
            maxRes = res
            maxResStr = resStr
        if res < minRes:
            minRes = res
            minResStr = resStr
        if i % 100000 == 0:
            print (str(i) + "/" + str(numRows))
    avgRes = sumRes / numRows
    return "min:" + minResStr + " max:" + maxResStr + " average resolution: " + str(avgRes)


if __name__ == "__main__":
    datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"

    if os.path.exists(csvPath):
        print("restoring csv data...")
        csvdata = pd.read_csv(csvPath)
    else:
        print("creating " + csvPath + "...")
        csvdata = createCsv(datasetDir, ageRange=[15, 100], minScore=0)
        csvdata.to_csv(csvPath, index=False)
        print(csvPath + " saved")

    if os.path.exists(indicesPath):
        print("restoring indices data...")
        file = open(indicesPath, "rb")
        indices = pickle.load(file)
    else:
        print("creating " + indicesPath + "...")
        indices = createIndices(csvdata)
        file = open(indicesPath, "wb")
        pickle.dump(indices, file)
        print(indicesPath + " saved")
    file.close()

    print ("finding image size range...")
    imgRange = findImageSizeRange(csvdata)
    print (imgRange)