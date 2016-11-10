from DataLoader import DataLoader, LoadFilesData

datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
csvPath = "./dataset.csv"
indicesPath = "./indices.p"

csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)
#run in new thread
loader = DataLoader(indices, csvdata, 1000)
loader.start()


