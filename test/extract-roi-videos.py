import cv2
import os
import json
import argparse
import numpy as np

CONTROL_MASK = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

def get_boundary(small, mask_analysis, ow, oh):
    ah, aw = small.shape[:2]
    scaled_w, scaled_h = int(aw * 0.4), int(ah * 0.4)

    overlay_positions = {
        'left_top': (aw // 2 - int(0.4 * aw / 2) // 2, ah - int(0.4 * ah / 2)),
        'left_bottom': (aw // 2 - int(0.4 * aw / 2) // 2, 0),
        'right_top_left': (aw - int(0.4 * aw / 4), ah - int(0.4 * ah / 2)),
        'right_bottom_left': (aw - int(0.4 * aw / 4), 0),
        'right_top_right': (0, ah - int(0.4 * ah / 2)),
        'right_bottom_right': (0, 0)
    }

    def extract_region(frame, pos, a):
        size = (scaled_w // (2 * a), scaled_h // 2)
        x, y = pos
        w, h = size
        return frame[y:y + h, x:x + w]

    left_top = extract_region(small, overlay_positions['left_top'], 1)
    left_bottom = extract_region(small, overlay_positions['left_bottom'], 1)
    right_top_left = extract_region(small, overlay_positions['right_top_left'], 2)
    right_bottom_left = extract_region(small, overlay_positions['right_bottom_left'], 2)
    right_top_right = extract_region(small, overlay_positions['right_top_right'], 2)
    right_bottom_right = extract_region(small, overlay_positions['right_bottom_right'], 2)

    left_half = np.vstack((left_top, left_bottom))
    right_top = np.hstack((right_top_left, right_top_right))
    right_bottom = np.hstack((right_bottom_left, right_bottom_right))
    right_half = np.vstack((right_top, right_bottom))

    full_scaled_mask = np.hstack((left_half, right_half))
    mask_full = cv2.resize(full_scaled_mask, (aw, ah), interpolation=cv2.INTER_LINEAR)

    # detect bright/colored overlay pixels (works for white and red markers)
    b, g, r = cv2.split(mask_full)
    bright = np.maximum(np.maximum(b, g), r) >= 128
    out = np.zeros_like(mask_analysis)
    out[bright & (mask_analysis == 255)] = 255

    k = max(2, int(5 * aw / ow + 0.5))
    kernel = np.ones((k, k), np.uint8)
    out_clean = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    mid = aw // 2

    def get_rectangle(part):
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(c)
            x0 = x * ow // aw
            y0 = y * oh // ah
            x1 = (x + w) * ow // aw + (1 if (x + w) * ow % aw else 0)
            y1 = (y + h) * oh // ah + (1 if (y + h) * oh % ah else 0)
            return [x0, y0, x1, y1]
        return None

    return {
        'left': get_rectangle(out_clean[:, :mid]),
        'right': get_rectangle(out_clean[:, mid:])
    }

parser = argparse.ArgumentParser(description="Extract ROI of AR Video")
parser.add_argument("filepath", type=str, help="ar video file path")
parser.add_argument("--border", type=int, nargs="*", default=[5], metavar="N", help="border px: 1 value for all sides, or 4 for LEFT TOP RIGHT BOTTOM (default: 5)")
parser.add_argument("--downscale", type=int, default=0, help="analysis downscale factor (default: auto: 2, or 4 for very large videos)")
parser.add_argument("--step", type=int, default=1, help="process every Nth frame (default: 1 = all frames)")
args = parser.parse_args()

b = args.border
if len(b) == 1:
    bl = bt = br = bb = b[0]
elif len(b) == 4:
    bl, bt, br, bb = b
else:
    parser.error("--border requires 1 or 4 values (left top right bottom)")

def choose_scale(w):
    if args.downscale:
        return args.downscale
    return 4 if w >= 6144 else 2

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
    scale = choose_scale(w)
    aw, ah = max(1, w // scale), max(1, h // scale)
    mask_analysis = cv2.resize(CONTROL_MASK, (aw, ah))
    small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
    area = get_boundary(small, mask_analysis, w, h)
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
        if not ret:
            break
        i += 1
        if i == 1:
            h, w = frame.shape[:2]
            scale = choose_scale(w)
            aw, ah = max(1, w // scale), max(1, h // scale)
            mask_analysis = cv2.resize(CONTROL_MASK, (aw, ah))
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
        if args.step > 1 and (i - 1) % args.step != 0:
            continue

        print("scan frame", i, "/", total_frames)
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        area = get_boundary(small, mask_analysis, w, h)
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
    for x in result['area']:
        x1,y1,x2,y2 = result['area'][x]
        w = x2-x1
        h=y2-y1
        _, ext = os.path.splitext(args.filepath)
        cmd = f"ffmpeg -i \"{args.filepath}\" -vf \"crop={w}:{h}:{x1}:{y1}\" -y \"{out_filename}_roi_{x}{ext}\""
        print(cmd)
        os.system(cmd)
        with open(f"{out_filename}_roi_{x}.json",'w') as f:
            json.dump({
                'x': x1,
                'y': y1,
                'w': w,
                'h': h
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