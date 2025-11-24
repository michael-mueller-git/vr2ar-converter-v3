import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture("input.mp4")
ret, frame = cap.read()

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
#cv2.imwrite("left_top.png", left_top)
left_bottom = extract_region(frame, overlay_positions['left_bottom'], 1)
#cv2.imwrite("left_bottom.png", left_bottom)
right_top_left = extract_region(frame, overlay_positions['right_top_left'], 2)
#cv2.imwrite("right_top_left.png", right_top_left )
right_bottom_left = extract_region(frame, overlay_positions['right_bottom_left'], 2)
#cv2.imwrite("right_bottom_left.png", right_bottom_left)
right_top_right = extract_region(frame, overlay_positions['right_top_right'], 2)
#cv2.imwrite("right_top_right.png", right_top_right)
right_bottom_right = extract_region(frame, overlay_positions['right_bottom_right'], 2)
#cv2.imwrite("right_bottom_right .png", right_bottom_right)

left_half = np.vstack((left_top, left_bottom))
#cv2.imwrite("left_half.png", left_half)
right_top = np.hstack((right_top_left, right_top_right))
#cv2.imwrite("right_top.png", right_top)
right_bottom = np.hstack((right_bottom_left, right_bottom_right))
#cv2.imwrite("right_bottom.png", right_bottom)
right_half = np.vstack((right_top, right_bottom))
#cv2.imwrite("right_half.png", right_half)



full_scaled_mask = np.hstack((left_half, right_half))
#cv2.imwrite("mask.png", full_scaled_mask)

original_mask = cv2.resize(full_scaled_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#cv2.imwrite("mask.png", original_mask)

original_mask = Image.fromarray(original_mask)

binary_mask = original_mask.convert("1")  # Pure black and white mask, mode "1"
#binary_mask.save("binary_mask.png")

if False:
    img_scaled = cv2.resize(frame, (binary_mask.size[1], binary_mask.size[0]))

    mask = Image.fromarray(binary_mask).convert("L")
    preview = Image.composite(
        Image.new("RGB", binary_mask.size, "blue"),
        Image.fromarray(img_scaled).convert("RGBA"),
        mask.point(lambda p: 100 if p > 1 else 0)
    )
    preview.save("result.png")

pil_mask = binary_mask

# Resize frame to match mask size (note Pillow size is width, height)
img_scaled = cv2.resize(frame, (pil_mask.size[0], pil_mask.size[1]))  # width, height order

# Convert binary_mask (Image) to grayscale mask for transparency
mask_gray = pil_mask.convert("L")

# Create composite preview: blue background and frame with mask
preview = Image.composite(
    Image.new("RGB", pil_mask.size, "blue"),
    Image.fromarray(img_scaled).convert("RGBA"),
    mask_gray.point(lambda p: 100 if p > 1 else 0)
)

preview.save("result.png")
