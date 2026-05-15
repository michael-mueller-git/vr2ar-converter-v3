import argparse
import glob
import os
import subprocess

import cv2


class FFmpegStream:
    @staticmethod
    def frame_to_timestamp(frame, fps):
        total_seconds = frame / fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True,
    )
    num, den = result.stdout.strip().split("/")
    fps = round(float(num) / float(den))

    return type("VideoInfo", (), {"width": width, "height": height, "fps": fps})()


def get_first_mask(mask_dir):
    pngs = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {mask_dir}")
    first = pngs[0]
    img = cv2.imread(first, cv2.IMREAD_UNCHANGED)
    h = img.shape[0]
    return first, h


def main():
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
        "--method",
        type=int,
        default=1,
        help="We have difrent methods implemented to merge chose one of the available [0,1]",
    )
    parser.add_argument(
        "--eq2fisheye",
        action="store_true",
        help="Convert equirectangular to fisheye projection",
    )
    parser.add_argument("--crf", type=int, default=16, help="libx265 CRF value")
    args = parser.parse_args()

    video = args.video
    mask_dir = args.mask_dir
    output_height = args.output_height
    eq2fisheye = args.eq2fisheye
    crf = args.crf
    method = args.method

    video_info = get_video_info(video)
    first_mask_path, mask_h = get_first_mask(mask_dir)
    mask_seq = os.path.join(mask_dir, "%06d.png")

    stem = os.path.splitext(os.path.basename(video))[0]
    result_name = f"{stem}-merged.mp4"

    out_resolution = f"{video_info.width}:{video_info.height}"
    scale = video_info.height / mask_h * 0.4
    if output_height > 0:
        out_w = int(output_height * 2)
        out_h = int(output_height)
        scale = out_h / mask_h * 0.4
        out_resolution = f"{out_w}:{out_h}"
        print("use custom output resolution", out_resolution)

    if method == 0:
        fc2 = f'"[1]scale=iw*{scale}:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h"'

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-thread_queue_size",
            "64",
            "-ss",
            FFmpegStream.frame_to_timestamp(0, video_info.fps),
            "-hwaccel",
            "auto",
            "-i",
            '"' + str(video) + '"',
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "-vcodec",
            "rawvideo",
            "-an",
            "-sn",
        ]

        if eq2fisheye:
            cmd += [
                "-filter_complex",
                f'"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack,scale={out_resolution}[v]"',
                "-map",
                "[v]",
            ]
        else:
            cmd += ["-filter_complex", f'"[0:v]scale={out_resolution}[v]"', "-map", "[v]"]

        cmd += [
            "-",
            "|",
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            out_resolution,
            "-r",
            str(video_info.fps),
            "-thread_queue_size",
            "64",
            "-i",
            "pipe:0",
            "-r",
            str(video_info.fps),
            "-thread_queue_size",
            "64",
            "-i",
            '"' + mask_seq + '"',
            "-i",
            '"' + first_mask_path + '"',
            "-r",
            str(video_info.fps),
            "-i",
            '"' + str(video) + '"',
            "-filter_complex",
            fc2,
            "-c:v",
            "libx265",
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-map",
            '"3:a:?"',
            "-c:a",
            "copy",
            '"' + result_name + '"',
            "-y",
        ]
    else if method == 1:
        fc = ""
        if eq2fisheye:
            fc += f"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; "
            fc += f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
            fc += f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
            fc += f"[leftfisheye][rightfisheye]hstack,scale={out_resolution}[bg]; "
        else:
            fc += f"[0:v]scale={out_resolution}[bg]; "

# 3. Strict Frame-to-Frame Sync Guarantee (Overrides VFR drift)
# This forces Frame 1 of Video to rigidly lock to Frame 1 of the Masks
        fc += f"[bg]setpts=N/FRAME_RATE/TB[bg_sync]; "
        fc += f"[1:v]setpts=N/FRAME_RATE/TB[mask_seq_sync]; "
        fc += f"[2:v]setpts=N/FRAME_RATE/TB[mask_static_sync]; "

# 4. Scale masks and apply the static 'mask.png'
        fc += f"[mask_seq_sync]scale=iw*{scale}:-1[alpha_scaled]; "
        fc += f"[mask_static_sync][alpha_scaled]scale2ref[mask_scaled][alpha_ref]; "
        fc += f"[alpha_ref][mask_scaled]alphamerge,split=2[masked_alpha1][masked_alpha2]; "

# 5. Split branches (FFmpeg 7.0+ handles buffering automatically, no 'fifo' needed)
        fc += f"[masked_alpha1]crop=iw/2:ih:0:0,split=2[l1][l2]; "
        fc += f"[masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[r1][r2][r3][r4]; "

# 6. Cascaded Overlays (eof_action=pass ensures completion if masks drop early)
        fc += f"[bg_sync][l1]overlay=W*0.5-w*0.5:-0.5*h:eof_action=pass[out_lt]; "
        fc += f"[out_lt][l2]overlay=W*0.5-w*0.5:H-0.5*h:eof_action=pass[out_tb]; "
        fc += f"[out_tb][r1]overlay=0-w*0.5:-0.5*h:eof_action=pass[out_l_lt]; "
        fc += f"[out_l_lt][r2]overlay=0-w*0.5:H-0.5*h:eof_action=pass[out_tb_ltb]; "
        fc += f"[out_tb_ltb][r3]overlay=W-w*0.5:-0.5*h:eof_action=pass[out_r_lt]; "
        fc += f"[out_r_lt][r4]overlay=W-w*0.5:H-0.5*h:eof_action=pass[out]"

# 7. Single FFmpeg Command Execution
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-hwaccel", "auto",
            
            # Input 0: The Original Video
            "-i", f'"{video}"',
            
            # Input 1: Mask image sequence (forced to matching video fps)
            "-framerate", str(video_info.fps),
            "-i", '"process/masks/%06d.png"',
            
            # Input 2: Static mask image (MUST loop to prevent stopping at frame 1)
            "-loop", "1",
            "-framerate", str(video_info.fps),
            "-i", '"mask.png"',
            
            # Apply unified filtergraph
            "-filter_complex", f'"{fc}"',
            
            # Map video from filtergraph, Map audio from Input 0
            "-map", '"[out]"',
            "-map", '"0:a?"',
            
            # Output Settings
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", "veryfast",
            "-c:a", "copy",

            # FIX: Explicitly specify the output framerate to prevent fallback to 25fps
            "-r", str(video_info.fps),

            f'"{result_name}"'
        ]

    print(" ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True)


if __name__ == "__main__":
    main()
