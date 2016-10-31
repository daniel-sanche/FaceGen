import os
from scipy.io import loadmat
import pandas as pd

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

age = dateOfBirth - yearTaken

dataTable = {"name":name, "birthday":dateOfBirth, "year_taken":yearTaken, "isMale":gender,
             "face_location":faceLocation, "face_score":faceScore, "second_face":faceScore2, "path":path}

df = pd.DataFrame(dataTable)
df.to_csv("./wiki.csv")

print (df)
