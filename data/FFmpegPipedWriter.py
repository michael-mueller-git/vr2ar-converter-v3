import sys
import os
import time
import glob
import argparse
import subprocess
import tempfile
import shutil
import cv2
import psutil
import shlex

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, ".", "data")

    if os.path.isdir(DATA_DIR) and DATA_DIR not in sys.path:
        sys.path.append(DATA_DIR)

    from ffmpegstream import VideoInfo, FFmpegStream
except:
    from data.ffmpegstream import VideoInfo, FFmpegStream


class FFmpegPipedWriter:
    def __init__(self, video, mask_dir, output_height, eq2fisheye, crf):
        self.video_path = video
        self.video_info = FFmpegStream.get_video_info(video)
        self.completed = False
        self.error = None
        self.mask_dir = mask_dir
        self.eq2fisheye = eq2fisheye
        self.crf = crf
        self.output_height = output_height

        self.out_w = self.video_info.width
        self.out_h = self.video_info.height
        if output_height > 0:
            self.out_w = int(output_height * 2)
            self.out_h = int(output_height)
            print("use custom output resolution", f"{self.out_w}:{self.out_h}")

        base, ext = os.path.splitext(os.path.basename(video))
        self.out_path = f"{base}-merged{ext}"
        self.result_name = f"{base}-alpha{ext}"

    def get_video_path(self):
        if not self.is_finished():
            return None
        return self.result_name

    def is_finished(self):
        return self.completed

    def set_batch(self, frame_end):
        pass

    def set_end(self):
        tmp_dir = None
        try:
            out_resolution_filter = f"{self.out_w}:{self.out_h}"
            out_resolution_raw = f"{self.out_w}x{self.out_h}"

            mask_files = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")))
            if not mask_files:
                raise FileNotFoundError(f"No mask PNGs found in {self.mask_dir}")
            first_mask = cv2.imread(mask_files[0], cv2.IMREAD_UNCHANGED)
            if first_mask is None:
                raise ValueError(f"Could not read mask file: {mask_files[0]}")
            mask_h = first_mask.shape[0]

            scale = self.video_info.height / mask_h * 0.4
            if self.output_height > 0:
                scale = self.out_h / mask_h * 0.4

            fc2 = (
                f"[1]scale=iw*{scale}:-1[alpha];"
                f"[2][alpha]scale2ref[mask][alpha];"
                f"[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; "
                f"[masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; "
                f"[masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; "
                f"[0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];"
                f"[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; "
                f"[out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];"
                f"[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; "
                f"[out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];"
                f"[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h"
            )

            mask_pattern = os.path.join(self.mask_dir, "%06d.png")

            ffmpeg1 = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "warning",
                "-thread_queue_size", "64",
                "-ss", FFmpegStream.frame_to_timestamp(0, self.video_info.fps),
                "-hwaccel", "auto",
                "-i", self.video_path,
                "-f", "image2pipe",
                "-pix_fmt", "bgr24",
                "-vsync", "passthrough",
                "-vcodec", "rawvideo",
                "-an",
                "-sn",
            ]
            if self.eq2fisheye:
                ffmpeg1 += [
                    "-filter_complex",
                    (
                        f"[0:v]split=2[left][right]; "
                        f"[left]crop=ih:ih:0:0[left_crop]; "
                        f"[right]crop=ih:ih:ih:0[right_crop]; "
                        f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
                        f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
                        f"[leftfisheye][rightfisheye]hstack,scale={out_resolution_filter}[v]"
                    ),
                    "-map", "[v]",
                ]
            else:
                ffmpeg1 += [
                    "-filter_complex",
                    f"[0:v]scale={out_resolution_filter}[v]",
                    "-map", "[v]",
                ]
            ffmpeg1 += ["-f", "rawvideo", "-"]

            ffmpeg2 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", out_resolution_raw,
                "-r", str(self.video_info.fps),
                "-thread_queue_size", "64",
                "-i", "pipe:0",
                "-r", str(self.video_info.fps),
                "-thread_queue_size", "64",
                "-framerate", str(self.video_info.fps),
                "-i", mask_pattern,
                "-i", "mask.png",
                "-r", str(self.video_info.fps),
                "-i", self.video_path,
                "-filter_complex", fc2,
                "-c:v", "libx265",
                "-crf", str(self.crf),
                "-preset", "veryfast",
                "-map", "3:a?",
                "-c:a", "copy",
                self.result_name,
                "-y",
            ]

            tmp_dir = tempfile.mkdtemp(prefix="ar_video_")
            script_path = os.path.join(tmp_dir, "pipeline.sh")
            decode_log = os.path.join(tmp_dir, "decode.log")
            encode_log = os.path.join(tmp_dir, "encode.log")

            cmd1_str = " ".join(shlex.quote(str(a)) for a in ffmpeg1)
            cmd2_str = " ".join(shlex.quote(str(a)) for a in ffmpeg2)

            script = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                f"{cmd1_str} 2>{shlex.quote(decode_log)} | \\\n"
                f"{cmd2_str} 2>{shlex.quote(encode_log)}\n"
            )

            with open(script_path, "w") as f:
                f.write(script)
            os.chmod(script_path, 0o755)

            print(f"Starting direct ffmpeg pipeline for {self.video_path}")
            self.process = subprocess.Popen(
                ["bash", script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            pid = self.process.pid
            proc = psutil.Process(pid)
            low_cpu_since = None
            CPU_THRESHOLD = 25.0
            CPU_TIMEOUT = 30

            while True:
                ret = self.process.poll()
                if ret is not None:
                    break

                cpu = proc.cpu_percent(interval=1)
                if cpu < CPU_THRESHOLD:
                    if low_cpu_since is None:
                        low_cpu_since = time.time()
                    elif time.time() - low_cpu_since >= CPU_TIMEOUT:
                        print(f"CPU stalled: below {CPU_THRESHOLD}% for {CPU_TIMEOUT} seconds (current: {cpu:.1f}%)")
                        self.process.kill()
                        self.process.wait()
                        self.error = f"CPU stalled: below {CPU_THRESHOLD}% for {CPU_TIMEOUT} seconds"
                        break
                else:
                    low_cpu_since = None

            if self.process.returncode is not None and self.process.returncode != 0:
                print(f"Pipeline failed with return code {self.process.returncode}")
                stderr_out = self.process.stderr.read() if self.process.stderr else ""
                if stderr_out:
                    print(stderr_out[:1000])
                for name, path in [("decode", decode_log), ("encode", encode_log)]:
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        with open(path) as f:
                            print(f"--- {name} stderr log ---")
                            print(f.read()[:1000])
                if not self.error:
                    self.error = f"ffmpeg returned {self.process.returncode}"
            else:
                print("Pipeline completed successfully")

        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            self.error = str(e)
        finally:
            if hasattr(self, 'process') and self.process is not None and self.process.returncode is None:
                self.process.kill()
                self.process.wait()
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self.completed = True


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
        "--batch",
        type=int,
        default=999999,
        help="Batch",
    )
    parser.add_argument(
        "--eq2fisheye",
        action="store_true",
        help="Convert equirectangular to fisheye projection",
    )
    parser.add_argument("--crf", type=int, default=16, help="libx265 CRF value")
    args = parser.parse_args()

    ar_writer = FFmpegPipedWriter(args.video, args.mask_dir, args.output_height, args.eq2fisheye, args.crf)
    ar_writer.set_batch(args.batch)
    ar_writer.set_end()
    while not ar_writer.is_finished():
        time.sleep(1)

    print("completed", ar_writer.get_video_path())
