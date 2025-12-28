import pickle
import argparse

parser = argparse.ArgumentParser(description="Edit a job pickle file")
parser.add_argument("pkl_path", type=str, help="Path to the pickle (.pkl) file")
parser.add_argument("height", type=int, help="New Video output Height as an integer")
args = parser.parse_args()

job = pickle.load(args.pkl_path)
job["outputHeight"] = args.height
pickle.dump(job, args.pkl_path)

