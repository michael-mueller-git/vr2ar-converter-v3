import cv2
import os
import json
import argparse
import math
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

parser = argparse.ArgumentParser(description="Extract ROI of VR Video as Regular View")
parser.add_argument("filepath", type=str, help="ar video file path")
parser.add_argument("--border", type=int, nargs="*", default=[5], metavar="N", help="border px: 1 value for all sides, or 4 for LEFT TOP RIGHT BOTTOM (default: 5)")
parser.add_argument("--source", type=str, choices=["fisheye", "equirect"], required=True, help="source projection of the video")
parser.add_argument("--fov", type=float, default=180, help="fisheye lens fov in degrees (default: 180)")
args = parser.parse_args()

b = args.border
if len(b) == 1:
    bl = bt = br = bb = b[0]
elif len(b) == 4:
    bl, bt, br, bb = b
else:
    parser.error("--border requires 1 or 4 values (left top right bottom)")

def compute_projection(x1, y1, x2, y2, half_x, half_w, half_h, frame_w, frame_h):
    rw = x2 - x1
    rh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if args.source == "fisheye":
        r_max = math.hypot(half_w, half_h) / 2
        hf = args.fov / 2
        cdx = cx - half_x - half_w / 2
        cdy = cy - half_h / 2

        def dir(dx, dy):
            r = math.hypot(dx, dy)
            if r == 0:
                return 0.0, 0.0
            theta = math.radians(r / r_max * hf)
            yaw = math.degrees(math.atan2(math.sin(theta) * dx / r, math.cos(theta)))
            pitch = -math.degrees(math.asin(math.sin(theta) * dy / r))
            return yaw, pitch

        yaw, pitch = dir(cdx, cdy)
        yaw_l, _ = dir(cdx - rw / 2, cdy)
        yaw_r, _ = dir(cdx + rw / 2, cdy)
        _, pitch_t = dir(cdx, cdy - rh / 2)
        _, pitch_b = dir(cdx, cdy + rh / 2)
        h_fov = min(abs(yaw_r - yaw_l), 170)
        v_fov = min(abs(pitch_t - pitch_b), 170)
        eq_w = half_w
        eq_h = half_h // 2
    else:
        h_fov = rw / frame_w * 360
        v_fov = rh / frame_h * 180
        yaw = cx / frame_w * 360 - 180
        pitch = -(cy / frame_h * 180 - 90)
        eq_w = frame_w
        eq_h = frame_h
    ow = rw if rw % 2 == 0 else rw - 1
    oh = rh if rh % 2 == 0 else rh - 1
    return {
        'type': args.source,
        'fov': args.fov if args.source == "fisheye" else 0,
        'yaw': yaw,
        'pitch': pitch,
        'h_fov': h_fov,
        'v_fov': v_fov,
        'out_w': ow,
        'out_h': oh,
        'half_x': half_x,
        'half_w': half_w,
        'half_h': half_h,
        'frame_w': frame_w,
        'frame_h': frame_h,
        'eq_w': eq_w,
        'eq_h': eq_h
    }

def is_image_input():
    return args.filepath.endswith(".jpg") or args.filepath.endswith(".png")

result = {}
if is_image_input():
    frame = cv2.imread(args.filepath)
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
    area = get_boundary(frame)
    print(area)
    for x in result['area']:
        if area[x] is not None:
            result['area'][x][0] = min((result['area'][x][0], area[x][0]))
            result['area'][x][1] = min((result['area'][x][1], area[x][1]))
            result['area'][x][2] = max((result['area'][x][2], area[x][2]))
            result['area'][x][3] = max((result['area'][x][3], area[x][3]))

else:
    cap = cv2.VideoCapture(args.filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
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
    result['area'][x][0] = max((result['area'][x][0] - bl, 0))
    result['area'][x][1] = max((result['area'][x][1] - bt, 0))
    result['area'][x][2] = min((result['area'][x][2] + br, result['size']['w']))
    result['area'][x][3] = min((result['area'][x][3] + bb, result['size']['h']))

if result_valid:
    print(result)
    out_filename = os.path.splitext(os.path.basename(args.filepath))[0]
    frame_w = result['size']['w']
    frame_h = result['size']['h']
    half_w = frame_w // 2
    half_h = frame_h
    for x in result['area']:
        x1,y1,x2,y2 = result['area'][x]
        w = x2-x1
        h=y2-y1
        half_x = 0 if x == 'left' else half_w
        proj = compute_projection(x1, y1, x2, y2, half_x, half_w, half_h, frame_w, frame_h)
        if args.source == "fisheye":
            vf = (f"crop={half_w}:{half_h}:{half_x}:0,"
                  f"v360=input=fisheye:output=flat:yaw={proj['yaw']:.2f}:pitch={proj['pitch']:.2f}:"
                  f"h_fov={proj['h_fov']:.2f}:v_fov={proj['v_fov']:.2f}:w={proj['out_w']}:h={proj['out_h']}:id_fov={args.fov},"
                  f"format=yuv420p")
        else:
            vf = (f"v360=input=equirect:output=flat:yaw={proj['yaw']:.2f}:pitch={proj['pitch']:.2f}:"
                  f"h_fov={proj['h_fov']:.2f}:v_fov={proj['v_fov']:.2f}:w={proj['out_w']}:h={proj['out_h']},"
                  f"format=yuv420p")
        _, ext = os.path.splitext(args.filepath)
        cmd = f"ffmpeg -i \"{args.filepath}\" -vf \"{vf}\" -y \"{out_filename}_roi_{x}{ext}\""
        print(cmd)
        os.system(cmd)
        with open(f"{out_filename}_roi_{x}.json",'w') as f:
            json.dump({
                'x': x1,
                'y': y1,
                'w': w,
                'h': h,
                'projection': proj
            }, f, indent=4)

    lw, lh = result['area']['left'][2] - result['area']['left'][0], result['area']['left'][3] - result['area']['left'][1]
    rw, rh = result['area']['right'][2] - result['area']['right'][0], result['area']['right'][3] - result['area']['right'][1]
    total_w = lw + rw
    total_h = max(lh, rh)

    if lh == rh:
        filter_complex = "[0:v]format=yuv420p[l];[1:v]format=yuv420p[r];[l][r]hstack=inputs=2[v]"
    elif lh < rh:
        filter_complex = (f"[0:v]format=yuv420p,pad={lw}:{total_h}:0:0:black[l];"
                          f"[1:v]format=yuv420p[r];[l][r]hstack=inputs=2[v]")
    else:
        filter_complex = (f"[0:v]format=yuv420p[l];"
                          f"[1:v]format=yuv420p,pad={rw}:{total_h}:0:0:black[r];[l][r]hstack=inputs=2[v]")

    cmd = (f"ffmpeg -i \"{out_filename}_roi_left{ext}\" -i \"{out_filename}_roi_right{ext}\" "
           f"-filter_complex \"{filter_complex}\" -map \"[v]\" -c:v libx264 -preset fast -crf 18 -y "
           f"\"{out_filename}_roi_sbs{ext}\"")
    print(cmd)
    os.system(cmd)

    with open(f"{out_filename}_roi_left.json") as f:
        left_source = json.load(f)
    with open(f"{out_filename}_roi_right.json") as f:
        right_source = json.load(f)

    with open(f"{out_filename}_roi_sbs.json", 'w') as f:
        json.dump({
            'size': {
                'w': total_w,
                'h': total_h
            },
            'views': {
                'left': {'x': 0, 'y': 0, 'w': lw, 'h': lh, 'source': left_source},
                'right': {'x': lw, 'y': 0, 'w': rw, 'h': rh, 'source': right_source}
            }
        }, f, indent=4)

else:
    print("invalid result")
