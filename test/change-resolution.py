import pickle
import argparse
import os
import cv2

def resize_masks_keep_ratio(masks, new_height):
    for m in masks:
        h, w = m['frameL'].shape[:2]
        scale = new_height / float(h)
        new_width = int(w * scale)

        m['frameL']     = cv2.resize(m['frameL'],     (new_width, new_height))
        m['frameR']     = cv2.resize(m['frameR'],     (new_width, new_height))
        m['frameLGray'] = cv2.resize(m['frameLGray'], (new_width, new_height))
        m['frameRGray'] = cv2.resize(m['frameRGray'], (new_width, new_height))
        m['maskL'] = m['maskL'].resize((new_width, new_height))
        m['maskR'] = m['maskR'].resize((new_width, new_height))
    return masks


parser = argparse.ArgumentParser(description="Edit a job pickle file")
parser.add_argument("pkl_path", type=str, help="Path to the pickle (.pkl) file")
parser.add_argument("out_height", type=int, help="New Video output Height as an integer")
parser.add_argument("mask_height", type=int, help="New Mask Height as an integer")
args = parser.parse_args()

with open(args.pkl_path, "rb") as f:
    job = pickle.load(f)

job["outputHeight"] = args.out_height
job["name"] = str(args.mask_height) + str(args.out_height) + "_" + job["name"]
job["masks"] = resize_masks_keep_ratio(job["masks"], args.mask_height)

with open(f"{args.mask_height}_{args.out_height}_" + str(os.path.basename(args.pkl_path)), "wb") as f:
    pickle.dump(job, f)

