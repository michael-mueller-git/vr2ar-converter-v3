import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Edit a job pickle file")
parser.add_argument("pkl_path", type=str, help="Path to the pickle (.pkl) file")
parser.add_argument("height", type=int, help="New Video output Height as an integer")
args = parser.parse_args()

with open(args.pkl_path, "rb") as f:
    job = pickle.load(f)
job["outputHeight"] = args.height
job["name"] = str(args.height) + "_" + job["name"]

with open(f"{args.height}_" + str(os.path.basename(args.pkl_path)), "wb") as f:
    pickle.dump(job, f)

