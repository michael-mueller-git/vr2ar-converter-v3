import argparse
import json
import os
import cv2

parser = argparse.ArgumentParser(description="Merge Upscaled ROI into AR Video")
parser.add_argument("src_video", type=str, help="ar src video file path")
parser.add_argument("left_roi", type=str, help="left upscaled roi")
parser.add_argument("right_roi", type=str, help="right upscaled roi")
parser.add_argument("--height", type=int, default=4096, help="Target Video Resolution Height")
args = parser.parse_args()

left_config = os.path.splitext(os.path.basename(args.left_roi))[0]
with open(f"{left_config}.json", "r") as f:
    roi_left = json.load(f)

right_config = os.path.splitext(os.path.basename(args.right_roi))[0]
with open(f"{right_config}.json", "r") as f:
    roi_right = json.load(f)

def get_resolution(video):
    cap = cv2.VideoCapture(args.src_video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return (height, width)

src_res = get_resolution(args.src_video)
scaling = args.height / src_res[0]
out_res = (round(src_res[0] * scaling), round(src_res[1] * scaling))

def get_out_resolution(roi_info, roi_video):
    h_old, w_old = roi_info["h"], roi_info["w"]
    h_new, w_new = get_resolution(roi_video)
    h_scaled = h_new / h_old
    w_scaled = w_new / w_old
    h_out = round(h_new / h_scaled * scaling)
    w_out = round(w_new / w_scaled * scaling)
    return (h_out, w_out)

left_out_res = get_out_resolution(roi_left, args.left_roi)
right_out_res = get_out_resolution(roi_right, args.right_roi)

left_out_pos = (round(roi_left["x"] * scaling), round(roi_left["y"] * scaling))
right_out_pos = (round(roi_right["x"] * scaling), round(roi_right["y"] * scaling))

out_name = os.path.splitext(os.path.basename(args.src_video))[0]

_, ext = os.path.splitext(args.src_video)

if ext == ".jpg":
    # improve output quality
    ext = ".png"

print("left res", left_out_res)
print("right res", right_out_res)

audio = "" if any(x == ext for x in [".png", ".jpg"]) else "-map 0:a -c:a copy"

cmd = f"ffmpeg -i \"{args.src_video}\" -i \"{args.left_roi}\" -i \"{args.right_roi}\" -filter_complex \""
cmd += f"[0:v]scale={out_res[1]}:{out_res[0]}[bg];"
cmd += f"[1:v]scale={left_out_res[1]}:{left_out_res[0]}[ol];"
cmd += f"[2:v]scale={right_out_res[1]}:{right_out_res[0]}[or];"
cmd += f"[bg][ol]overlay={left_out_pos[0]}:{left_out_pos[1]}:format=auto[tmp];"
cmd += f"[tmp][or]overlay={right_out_pos[0]}:{right_out_pos[1]}:format=auto[out]"
cmd += f"\"  -map \"[out]\" {audio} \"{out_name}-merged{ext}\""

print(cmd)
os.system(cmd)
