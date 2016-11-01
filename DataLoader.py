import os
from scipy.io import loadmat
from scipy.misc import imread, imresize
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle


def createCsv(datasetDir, outPath="./dataset.csv", ageRange=[15, 100], minScore=0, filterGender=True):
    #if file exists, read from disk instead of generating
    if os.path.exists(outPath):
        print("restoring saved csv...")
        return pd.read_csv(outPath)
    else:
        combinedDf = None
        for fileType in ["wiki", "imdb"]:
            matFile = loadmat(os.path.join(datasetDir, fileType+"_crop", fileType+".mat"))
            dateOfBirth = matFile[fileType]["dob"][0][0][0]
            yearTaken = matFile[fileType]["photo_taken"][0][0][0]
            path = matFile[fileType]["full_path"][0][0][0]
            gender = matFile[fileType]["gender"][0][0][0]
            name = matFile[fileType]["name"][0][0][0]
            faceLocation = matFile[fileType]["face_location"][0][0][0]
            faceScore = matFile[fileType]["face_score"][0][0][0]
            faceScore2 = matFile[fileType]["second_face_score"][0][0][0]

            birthYear = np.zeros(dateOfBirth.shape)
            age = np.zeros(dateOfBirth.shape)

            for i in range(0, dateOfBirth.shape[0]):
                # add age/birth year
                matlabBD = dateOfBirth[i]
                if matlabBD < 366:
                    matlabBD = 400

                pythonBd = datetime.fromordinal(int(matlabBD)) + timedelta(days=int(matlabBD) % 1) - timedelta(days=366)
                birthYear[i] = pythonBd.year
                age[i] = yearTaken[i] - pythonBd.year
                # fix name
                nameArr = name[i]
                if (nameArr.shape[0] > 0):
                    name[i] = nameArr[0].replace(",", "")
                else:
                    name[i] = ""
                # fix path
                pathArr = path[i]
                path[i] = os.path.join(datasetDir, fileType + "_crop", pathArr[0])

            dataTable = {"name": name, "age": age, "birthday": birthYear, "year_taken": yearTaken, "isMale": gender,
                         "face_location": faceLocation, "face_score": faceScore, "second_face": faceScore2, "path": path}
            # remove bad data
            df = pd.DataFrame(dataTable)
            if combinedDf is None:
                combinedDf = df
            else:
                combinedDf = pd.concat([combinedDf, df])
        numLeft = len(combinedDf.index)
        print(numLeft, " images found")
        if minScore is not None:
            combinedDf = combinedDf[combinedDf.face_score > minScore]
            numLeft = len(combinedDf.index)
            print("filtered low quality: ", numLeft, " images remaining")
        if ageRange is not None:
            combinedDf = combinedDf[combinedDf.age > ageRange[0]]
            combinedDf = combinedDf[combinedDf.age < ageRange[1]]
            numLeft = len(combinedDf.index)
            print("filtered bad ages: ", numLeft, " images remaining")
        if filterGender:
            combinedDf = combinedDf[combinedDf.isMale.notnull()]
            numLeft = len(combinedDf.index)
            print("filtered null sex: ", numLeft, " images remaining")
        if outPath is not None:
            print("saving...")
            combinedDf.to_csv(outPath, index=False)
        return combinedDf


def getStats(dataframe, ageRange, outPath="stats.csv"):
    numRows = len(dataframe.index)
    numAges = ageRange[1] - ageRange[0] + 1
    resultsArr = np.zeros([numAges, 2], dtype=int)
    for i in range(numRows):
        sex = int(dataframe["isMale"][i])
        age = int(dataframe["age"][i])
        adjustedAge = age - ageRange[0]
        resultsArr[adjustedAge, sex] = resultsArr[adjustedAge, sex] + 1
    df = pd.DataFrame(resultsArr, columns=["female", "male"], index=np.arange(ageRange[0], ageRange[1]+1))
    df.to_csv(outPath)
    return df

def createIndices(dataframe, ageRangeLimits=[30, 40, 50, 60, 70, 80, 101], fileName="indices.p"):
    if os.path.exists(fileName):
        file = open(fileName, "rb")
        print("restoring saved indices...")
        return  pickle.load(file)

    numRows = len(dataframe.index)
    menArr = [[] for x in ageRangeLimits]
    womenArr = [[] for x in ageRangeLimits]
    for i in range(numRows):
        male = bool(dataframe["isMale"][i])
        age = int(dataframe["age"][i])
        binNum = 0
        for binLimit in ageRangeLimits:
            if age < binLimit:
                break
            else:
                binNum = binNum + 1
        if male:
            menArr[binNum] += [i]
        else:
            womenArr[binNum] += [i]
    resultDict = {"Men":menArr, "Women":womenArr, "AgeBinLimits":ageRangeLimits}
    file = open( fileName, "wb" )
    pickle._dump(resultDict, file)
    return  resultDict


datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"

dataframe = createCsv(datasetDir, ageRange=[15, 100], minScore=0)
indices = createIndices(dataframe)





