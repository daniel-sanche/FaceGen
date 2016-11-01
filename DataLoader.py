import os
from scipy.io import loadmat
from scipy.misc import imread, imresize
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle
import time
import datetime
""""
creates a csv file containing information on all the faces
uses the information from the dataset's .mat files, and applies filtering to keep only good quality data

Params
    datasetDir: the directory of the IMDBWIKI dataset on the computer's hard drive
    agetRange: a vector containing the min and max age to keep. Helps trim out outlier errors in the dataset
    minScore: the minimum face score to keep. Removes bad quality data
    filterGender: a bool that determines whether to trim out faces with unlabeled geneders

Returns
    0: the dataframe the .csv represents
"""
def createCsv(datasetDir, outPath="./dataset.csv", ageRange=[15, 100], minScore=0, filterGender=True):
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
    return combinedDf

"""
creates a csv file detailing the age/gender breakdown of the csv dataset

Params
    csvdata:    a pandas dataframe from the csv file we are using as out dataset
    ageRange:   the start and end values for the ages we are using
    outPath:    the path to save the stats csv at

Returns
    0: a pandas dataframe representing the results
"""
def getStats(csvdata, ageRange=[15, 100], outPath="stats.csv"):
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

"""
creates files that contain a list of indices for each category we are training on.
returns a dictionary with 3 keys: "Men", "Women" and "AgeBunLimits"
AgeBinLimits contains a list of cut-off points that define each age range
Men and Women contains a list of lists, where each element represents an age bin,
and contains a list of indices of images that fall into that bin

Params
    csvdata:      the dataframe of the .csv file of good quality faces we are working with
    ageRangeLimits: a vector describing all the age ranges we are breaking the data into
                    each item describes the ages < this value that will belong in this bin

Returns:
    0:  a dictionary containing the indices
"""
def createIndices(csvdata, ageRangeLimits=[30, 40, 50, 60, 70, 80, 101]):
    numRows = len(csvdata.index)
    menArr = [[] for x in ageRangeLimits]
    womenArr = [[] for x in ageRangeLimits]
    for i in range(numRows):
        male = bool(csvdata["isMale"][i])
        age = int(csvdata["age"][i])
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
    return  resultDict

"""
Helper function to extract the next set of indices from the requested bin
Always returns numRequested indices, and handles looping back to the
beginning of the bin if necessary

Params
    binList:    the list of indices in this bin
    offset:     the offset to start grabbing indices from
    numRequested:   the number of indices to return

Returns
    0:  the list of indices extracted from the bin
    1:  the new offset we ended at
    2:  whether we passed over the end of the bin
"""
def _getFromBin(binList, offset, numRequested):
    startPt = offset
    endPt = min(startPt + numRequested, len(binList))
    returnList = binList[startPt:endPt]
    length = endPt - startPt
    looped = False
    while length < numRequested:
        looped = True
        startPt = 0
        endPt = min(numRequested - length, len(binList))
        returnList += binList[startPt:endPt]
        length = len(returnList)

    return returnList, endPt, looped

"""
Extracts a batch of images from the data. If the previous state is stored and
returned, the function can be called again to iterate through the data in batches

Params
    indices:    the indices dict for the data
    csvdata:    the pandas dataframe from the .csv of faces we are using
    batchSize:  the size of the batch we want to extract
    imageSize:  the size of the images to extract
    prevState:  the state containing the last indices we extracted, so we can get the next batch

Returns
    0:  a dictionary containing a vector for all the images (batchSize x imageSize),
        a vector of the ages (batchSize x 1), and a vector of the sexes (batchSize x 1) for the batch
    1:  the new state, whcih can be passed back in to get the next batch
    2:  a bool indicating whether we have visited all images at least one (since the start of the state)
"""
def getBatch(indices, csvdata, batchSize=1000, imageSize=[250, 250, 3], prevState=None):
    ageBins = indices["AgeBinLimits"]
    numBins = len(ageBins)
    numPerCat = int(round(batchSize / (numBins * 2), 0))
    if prevState is None:
        prevState = np.zeros([numBins, 2, 2], dtype=int)
    batchIndices = np.zeros([numPerCat * numBins * 2], dtype=int)
    menLists = indices["Men"]
    womenLists = indices["Women"]
    lastIdx = 0
    for i in range(numBins):
        newMen, newOffset, didLoop = _getFromBin(menLists[i], prevState[i, 1, 0], numPerCat)
        batchIndices[lastIdx:lastIdx+numPerCat] = newMen
        prevState[i, 1, 0] = newOffset
        if didLoop:
            prevState[i, 1, 1] = 1
        lastIdx = lastIdx+numPerCat
        newWomen, newOffset, didLoop = _getFromBin(womenLists[i], prevState[i, 0, 0], numPerCat)
        batchIndices[lastIdx:lastIdx + numPerCat] = newWomen
        prevState[i, 0, 0] = newOffset
        if didLoop:
            prevState[i, 1, 0] = 1
        lastIdx = lastIdx + numPerCat
    imageArr = np.zeros([batchSize]+imageSize, dtype=int)
    sexArr = np.zeros([batchSize, 1], dtype=bool)
    ageArr = np.zeros([batchSize, 1], dtype=int)
    i = 0
    for idx in batchIndices:
        path = csvdata["path"][idx]
        age = csvdata["age"][idx]
        sex = csvdata["isMale"][idx]
        image = imread(path)
        if image.shape != imageSize:
            image = imresize(image, imageSize)
        if len(image.shape) == 2:
            image = np.resize(image, imageSize)
        imageArr[i,:,:] = image
        sexArr[i] = sex
        ageArr[i] = age
        i = i + 1
    didVisitAll = np.sum(prevState[:,:,1]) == numBins * 2
    return {"image":imageArr, "sex":sexArr, "age":ageArr}, prevState, didVisitAll


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

    if os.path.exists(indicesPath):
        print("restoring indices data...")
        file = open(indicesPath, "rb")
        indices = pickle.load(file)
    else:
        print("creating " + indicesPath + "...")
        indices = createIndices(csvdata)
        file = open(indicesPath, "wb")
        pickle.dump(indices, file)
    file.close()


    offset = None
    didFinish = False
    i=0
    while not didFinish:
        start = time.time()
        batchData, offset, didFinish = getBatch(indices, csvdata, prevState=offset)
        end = time.time()
        diff = end - start
        finishedNum = np.sum(offset[:,:,1])
        print("finished " +str(i) + " :" + str(diff) + " ( " + str(finishedNum) +"/" + str(offset.shape[0] * offset.shape[1]) + " finished)")
        i = i + 1






