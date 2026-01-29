import cv2
import os
import json
import argparse
import numpy as np
from PIL import Image

CONTROL_MASK = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

def get_boundary(frame):
    h, w = frame.shape[:2]
    scaled_w, scaled_h = int(w * 0.4), int(h * 0.4)

    overlay_positions = {
        'left_top': (w // 2 - int(0.4 * w / 2) // 2, h - int(0.4 * h / 2)),
        'left_bottom': (w // 2 - int(0.4 * w / 2) // 2, 0),
        'right_top_left': (w - int(0.4 * w / 4), h - int(0.4 * h / 2)),
        'right_bottom_left': (w - int(0.4 * w / 4), 0),
        'right_top_right': (0, h - int(0.4 * h / 2)),
        'right_bottom_right': (0, 0)
    }

    def extract_region(frame, pos, a):
        size=(scaled_w//(2*a), scaled_h//2)
        x, y = pos
        w, h = size
        return frame[y:y+h, x:x+w]

    left_top = extract_region(frame, overlay_positions['left_top'], 1)
    left_bottom = extract_region(frame, overlay_positions['left_bottom'], 1)
    right_top_left = extract_region(frame, overlay_positions['right_top_left'], 2)
    right_bottom_left = extract_region(frame, overlay_positions['right_bottom_left'], 2)
    right_top_right = extract_region(frame, overlay_positions['right_top_right'], 2)
    right_bottom_right = extract_region(frame, overlay_positions['right_bottom_right'], 2)

    left_half = np.vstack((left_top, left_bottom))
    right_top = np.hstack((right_top_left, right_top_right))
    right_bottom = np.hstack((right_bottom_left, right_bottom_right))
    right_half = np.vstack((right_top, right_bottom))

    full_scaled_mask = np.hstack((left_half, right_half))
    original_mask = cv2.resize(full_scaled_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    original_mask = Image.fromarray(original_mask)
    binary_mask = original_mask.convert("1")  # Pure black and white mask
    binary_mask = np.array(binary_mask, dtype=np.uint8) * 255
    control_mask = cv2.resize(CONTROL_MASK, (binary_mask.shape[1], binary_mask.shape[0]))

    white1 = binary_mask == 255
    white2 = control_mask == 255
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[white1 & white2] = 255

    _, out_bin = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)

    # removes small white artifacts
    out_clean = cv2.morphologyEx(out_bin, cv2.MORPH_OPEN, kernel)

    height, width = out_clean.shape[:2]
    mid = width // 2

    def get_rectangle(out_clean):
        contours, _ = cv2.findContours(out_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(c)
            return [x,y,x+w,y+h]
        return None

    out_clean_left = out_clean[:, :mid]
    out_clean_right = out_clean[:, mid:]

    return {
        'left': get_rectangle(out_clean_left),
        'right': get_rectangle(out_clean_right)
    }

parser = argparse.ArgumentParser(description="Extract ROI of AR Video")
parser.add_argument("filepath", type=str, help="ar video file path")
parser.add_argument("--border", type=int, default=5, help="border value (default: 5)")
args = parser.parse_args()

cap = cv2.VideoCapture(args.filepath)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
i = 0
result = {}
while True:
    ret, frame = cap.read()
    if i == 0:
        h, w = frame.shape[:2]
        result = {
            'area': {
                'left': [w//2, h, 0, 0],
                'right': [w//2, h, 0, 0]
            },
            'size':  {
                'w': w,
                'h': h
            }
        }
    i += 1
    
    if not ret:
        break
        
    print("scan frame", i, "/", total_frames)
    area = get_boundary(frame)
    for x in result['area']:
        if area[x] is not None:
            result['area'][x][0] = min((result['area'][x][0], area[x][0]))
            result['area'][x][1] = min((result['area'][x][1], area[x][1]))
            result['area'][x][2] = max((result['area'][x][2], area[x][2]))
            result['area'][x][3] = max((result['area'][x][3], area[x][3]))

cap.release()

result_valid = True
for x in result['area']:
    if result['area'][x][0] > result['area'][x][2] or  result['area'][x][1] > result['area'][x][3]:
        result_valid = False

# apply offset of split
result['area']['right'][0] += (result['size']['w'] // 2)
result['area']['right'][2] += (result['size']['w'] // 2)

# apply custom border
for x in result['area']:
    result['area'][x][0] = max((result['area'][x][0] - args.border, 0))
    result['area'][x][1] = max((result['area'][x][1] - args.border, 0))
    result['area'][x][2] = min((result['area'][x][2] + args.border, result['size']['w']))
    result['area'][x][3] = min((result['area'][x][3] + args.border, result['size']['h']))

if result_valid:
    print(result)
    out_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    for x in result['area']:
        x1,y1,x2,y2 = result['area'][x]
        w = x2-x1
        h=y2-y1
        cmd = f"ffmpeg -i input.mkv -vf \"crop={w}:{h}:{x1}:{y1}\" -y {out_filename}_roi_{x}.mp4"
        print(cmd)
        os.system(cmd)
        with open(f"{out_filename}_roi_{x}.json",'w') as f:
            json.dump({
                'x': x1,
                'y': y1,
                'w': w,
                'h': h
            }, f, indent=4)

else:
    print("invalid result")
