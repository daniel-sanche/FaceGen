import os
from scipy.io import loadmat
from scipy.misc import imread, imresize
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def createCsv(datasetDir, outPath="./dataset.csv", ageRange=[15, 100], minScore=0):
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

    if minScore is not None:
        combinedDf = combinedDf[combinedDf.face_score > minScore]
    if ageRange is not None:
        combinedDf = combinedDf[combinedDf.age > ageRange[0]]
        combinedDf = combinedDf[combinedDf.age < ageRange[1]]
    if outPath is not None:
        combinedDf.to_csv(outPath, index=False)
    return combinedDf

def createCache(savePath="./images.npy", csvPath="./filtered.csv", imageSize=[257, 257, 3], limit=None):
    if os.path.exists(savePath):
        return np.load(savePath)
    else:
        if os.path.exists(csvPath):
            dataset = pd.read_csv(csvPath)
        else:
            dataset = createCsv(datasetDir, outPath=csvPath, ageRange=[15, 100], minScore=0)
        if limit is None:
            limit = len(dataset.index)
        imageArr = np.zeros([limit] + imageSize)
        for i in range(limit):
            path = dataset["path"][i]
            image = imread(path)
            if imageSize is not None:
                image = imresize(image, imageSize)
            if image.shape == imageSize:
                imageArr[i,:,:,:] = image
        np.save(savePath, imageArr)
        return imageArr



datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"

images = createCache()
print (images.shape)




