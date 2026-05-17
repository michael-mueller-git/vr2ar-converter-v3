import sys
import os
import cv2
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

if os.path.isdir(DATA_DIR) and DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from ffmpegstream import VideoInfo, FFmpegStream
from ArVideoWriter import ArVideoWriter


def main(video, mask_dir, output_height, eq2fisheye, crf):
    video_info = FFmpegStream.get_video_info(video)

    out_w = video_info.width
    out_h = video_info.height
    if output_height > 0:
        out_w = int(output_height * 2)
        out_h = int(output_height)
        print("use custom output resolution", f"{out_w}:{out_h}")

    config = {"parameter": {"width": out_w, "height": out_h}}
    if eq2fisheye:
        config["filter_complex"] = (
            f"[0:v]split=2[left][right]; "
            f"[left]crop=ih:ih:0:0[left_crop]; "
            f"[right]crop=ih:ih:ih:0[right_crop]; "
            f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
            f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
            f"[leftfisheye][rightfisheye]hstack,scale={out_w}:{out_h}[v]"
        )
    else:
        config["filter_complex"] = f"[0:v]scale={out_w}:{out_h}[v]"

    ffmpeg = FFmpegStream(video_path=video, config=config, skip_frames=0)

    base, ext = os.path.splitext(os.path.basename(video))
    out_path = f"{base}-merged{ext}"
    writer = ArVideoWriter(out_path, video_info.fps, crf=crf)

    i = 0
    while True:
        frame = ffmpeg.read()
        if frame is None:
            break

        print("process frame", i)
        mask_path = os.path.join(mask_dir, f"{i+1:06d}.png")
        if not os.path.exists(mask_path):
            break

        alpha = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        writer.add_frame(frame, alpha)
        i += 1

    ffmpeg.stop()
    writer.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge masks into AR video")
    parser.add_argument("video", type=str, help="Input video file path")
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="mask",
        help="Directory with mask PNG sequence (%%06d.png)",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=0,
        help="Target output height (0 = auto from video)",
    )
    parser.add_argument(
        "--eq2fisheye",
        action="store_true",
        help="Convert equirectangular to fisheye projection",
    )
    parser.add_argument("--crf", type=int, default=16, help="libx265 CRF value")
    args = parser.parse_args()

    main(args.video, args.mask_dir, args.output_height, args.eq2fisheye, args.crf)
