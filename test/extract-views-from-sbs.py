import os
import json
import argparse

parser = argparse.ArgumentParser(description="Extract original left/right views from SBS video using its JSON")
parser.add_argument("filepath", type=str, help="sbs video file path")
parser.add_argument("--json", dest="json_path", type=str, default=None, help="sbs json file path (default: <video>.json)")
args = parser.parse_args()

if args.json_path is None:
    args.json_path = os.path.splitext(args.filepath)[0] + ".json"

with open(args.json_path) as f:
    data = json.load(f)

out_filename = os.path.splitext(os.path.basename(args.filepath))[0]
_, ext = os.path.splitext(args.filepath)

for name, view in data['views'].items():
    x, y, w, h = view['x'], view['y'], view['w'], view['h']
    output = f"{out_filename}_view_{name}{ext}"
    cmd = f"ffmpeg -i \"{args.filepath}\" -vf \"crop={w}:{h}:{x}:{y}\" -y \"{output}\""
    print(cmd)
    os.system(cmd)
