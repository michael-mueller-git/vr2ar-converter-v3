import argparse
import json
import os
import cv2

parser = argparse.ArgumentParser(description="Merge Upscaled ROI into AR Video")
parser.add_argument("src_video", type=str, help="ar src video file path")
parser.add_argument("left_roi", type=str, nargs="?", help="left upscaled roi")
parser.add_argument("right_roi", type=str, nargs="?", help="right upscaled roi")
parser.add_argument("--sbs", type=str, help="combined sbs video instead of separate left/right rois")
parser.add_argument("--height", type=int, default=4096, help="Target Video Resolution Height")
args = parser.parse_args()

def load_config(video):
    config = os.path.splitext(os.path.basename(video))[0]
    with open(f"{config}.json", "r") as f:
        return json.load(f)

sbs_mode = args.sbs is not None
if sbs_mode:
    if args.left_roi is not None or args.right_roi is not None:
        parser.error("cannot combine --sbs with separate left/right roi arguments")
    sbs_info = load_config(args.sbs)
    roi_left = dict(sbs_info['views']['left']['source'])
    roi_right = dict(sbs_info['views']['right']['source'])
    sbs_left_rect = sbs_info['views']['left']
    sbs_right_rect = sbs_info['views']['right']
else:
    if args.left_roi is None or args.right_roi is None:
        parser.error("left_roi and right_roi are required unless --sbs is used")
    roi_left = load_config(args.left_roi)
    roi_right = load_config(args.right_roi)

def get_resolution(video):
    cap = cv2.VideoCapture(args.src_video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return (height, width)

src_res = get_resolution(args.src_video)
scaling = args.height / src_res[0]
out_res = (round(src_res[0] * scaling), round(src_res[1] * scaling))

left_out_pos = (round(roi_left["x"] * scaling), round(roi_left["y"] * scaling))
right_out_pos = (round(roi_right["x"] * scaling), round(roi_right["y"] * scaling))

out_name = os.path.splitext(os.path.basename(args.src_video))[0]

_, ext = os.path.splitext(args.src_video)

if ext == ".jpg":
    # improve output quality
    ext = ".png"

audio = "" if any(x == ext for x in [".png", ".jpg"]) else "-map 0:a -c:a copy"

def inverse_chain(proj, rect, scaling):
    ow = max(round(rect['w'] * scaling), 2)
    oh = max(round(rect['h'] * scaling), 2)
    if proj['type'] == "fisheye":
        fx = rect['x'] - proj['half_x']
        chain = (f"v360=input=flat:output=equirect:yaw={-proj['yaw']:.2f}:pitch={-proj['pitch']:.2f}:"
                 f"ih_fov={proj['h_fov']:.2f}:iv_fov={proj['v_fov']:.2f}:w={proj['eq_w']}:h={proj['eq_h']}:alpha_mask=1,"
                 f"v360=input=equirect:output=fisheye:d_fov={proj['fov']}:w={proj['half_w']}:h={proj['half_h']},"
                 f"crop={rect['w']}:{rect['h']}:{fx}:{rect['y']}")
    else:
        chain = (f"v360=input=flat:output=equirect:yaw={-proj['yaw']:.2f}:pitch={-proj['pitch']:.2f}:"
                 f"ih_fov={proj['h_fov']:.2f}:iv_fov={proj['v_fov']:.2f}:w={proj['frame_w']}:h={proj['frame_h']}:alpha_mask=1,"
                 f"crop={rect['w']}:{rect['h']}:{rect['x']}:{rect['y']}")
    return f"{chain},scale={ow}:{oh}"

if sbs_mode:
    cmd = f"ffmpeg -i \"{args.src_video}\" -i \"{args.sbs}\" -filter_complex \""
    cmd += f"[0:v]scale={out_res[1]}:{out_res[0]}[bg];"
    cmd += (f"[1:v]crop={sbs_left_rect['w']}:{sbs_left_rect['h']}:{sbs_left_rect['x']}:{sbs_left_rect['y']},"
            f"{inverse_chain(roi_left['projection'], roi_left, scaling)}[ol];")
    cmd += (f"[1:v]crop={sbs_right_rect['w']}:{sbs_right_rect['h']}:{sbs_right_rect['x']}:{sbs_right_rect['y']},"
            f"{inverse_chain(roi_right['projection'], roi_right, scaling)}[or];")
    cmd += f"[bg][ol]overlay={left_out_pos[0]}:{left_out_pos[1]}:format=auto[tmp];"
    cmd += f"[tmp][or]overlay={right_out_pos[0]}:{right_out_pos[1]}:format=auto[out]"
    cmd += f"\"  -map \"[out]\" {audio} \"{out_name}-merged{ext}\""
else:
    cmd = f"ffmpeg -i \"{args.src_video}\" -i \"{args.left_roi}\" -i \"{args.right_roi}\" -filter_complex \""
    cmd += f"[0:v]scale={out_res[1]}:{out_res[0]}[bg];"
    cmd += f"[1:v]{inverse_chain(roi_left['projection'], roi_left, scaling)}[ol];"
    cmd += f"[2:v]{inverse_chain(roi_right['projection'], roi_right, scaling)}[or];"
    cmd += f"[bg][ol]overlay={left_out_pos[0]}:{left_out_pos[1]}:format=auto[tmp];"
    cmd += f"[tmp][or]overlay={right_out_pos[0]}:{right_out_pos[1]}:format=auto[out]"
    cmd += f"\"  -map \"[out]\" {audio} \"{out_name}-merged{ext}\""

print(cmd)
os.system(cmd)
print(f"{out_name}-merged{ext}")
