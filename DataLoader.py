import os
from scipy.io import loadmat
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
#for root, dirs, files in os.walk(datasetDir):
#    for file in files:
#        if file.endswith(".mat"):
#            matFile = loadmat(os.path.join(root,file))
#            print (matFile)
matFile = loadmat(os.path.join(datasetDir, "wiki_crop/wiki.mat"))
dateOfBirth = matFile["wiki"]["dob"][0][0][0]
yearTaken = matFile["wiki"]["photo_taken"][0][0][0]
path = matFile["wiki"]["full_path"][0][0][0]
gender = matFile["wiki"]["gender"][0][0][0]
name = matFile["wiki"]["name"][0][0][0]
faceLocation = matFile["wiki"]["face_location"][0][0][0]
faceScore =  matFile["wiki"]["face_score"][0][0][0]
faceScore2 =  matFile["wiki"]["second_face_score"][0][0][0]

birthYear = np.zeros(dateOfBirth.shape)
age = np.zeros(dateOfBirth.shape)

for i in range(0, dateOfBirth.shape[0]):
    #add age/birth year
    matlabBD = dateOfBirth[i]
    pythonBd = datetime.fromordinal(int(matlabBD)) + timedelta(days=int(matlabBD) % 1) - timedelta(days=366)
    birthYear[i] = pythonBd.year
    age[i] = yearTaken[i] - pythonBd.year
    #fix name
    nameArr = name[i]
    if(nameArr.shape[0] >0):
        name[i] = nameArr[0].replace(",", "")
    else:
        name[i] = ""
    #fix path
    pathArr = path[i]
    path[i] = pathArr[0]


dataTable = {"name":name, "age":age, "birthday":birthYear, "year_taken":yearTaken, "isMale":gender,
             "face_location":faceLocation, "face_score":faceScore, "second_face":faceScore2, "path":path}

df = pd.DataFrame(dataTable)
df.to_csv("./wiki.csv", index=False)

