import sys
import os
import pandas as pd
import pickle


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
