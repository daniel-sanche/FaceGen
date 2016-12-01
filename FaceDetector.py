import cv2
from DataLoader import LoadFilesData, DataLoader
import numpy as np
from Visualization import visualizeImages
import glob

def detectedFace(image, cascadePath="./cascades"):
    for thisCascade in glob.glob(cascadePath+"/*.xml"):
        #convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #perform opencv face detection
        faceCascade = cv2.CascadeClassifier(thisCascade)
        faces = faceCascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=0,
            minSize=(32, 32),
            maxSize=(64, 64),
        )
        if len(faces) > 0:
            return True
    return False

if __name__ == "__main__":
    datasetDir = "/home/sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    loader = DataLoader(indices, csvdata, numPerBin=10, imageSize=64, numWorkerThreads=10, bufferMax=20,
                        debugLogs=False, useCached=False)
    loader.start()

    batchDict = loader.getData()
    imageSet = batchDict["image"]
    # convert to 8 bit int
    imageSet = ((imageSet + 1) * (255 / 2)).astype(np.uint8)

    numFound = 0
    errMat = np.zeros_like(imageSet)
    for i in range(imageSet.shape[0]):
        thisImage = imageSet[i, :,:,:]
        foundFace = detectedFace(thisImage)
        if not foundFace:
            errMat[numFound, :, : :] = thisImage
            numFound = numFound + 1
    print ("Num Error Images Found: " + str(numFound) + "/" + str(imageSet.shape[0]))
    if numFound > 0:
        visualizeImages(errMat[:numFound, :, :, :], numRows=1, fileName="errorImages.png")


