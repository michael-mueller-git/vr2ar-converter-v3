import sys
import os
import cv2
import time
import argparse
import subprocess
from threading import Thread
from queue import Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, ".", "data")

if os.path.isdir(DATA_DIR) and DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from ffmpegstream import VideoInfo, FFmpegStream
from ArVideoWriter import ArVideoWriter


class ArVideoWriterWrapper:
    def __init__(self, video, mask_dir, output_height, eq2fisheye, crf):
        self.video_path = video
        self.video_info = FFmpegStream.get_video_info(video)
        self.queue = Queue(maxsize=4096)
        self.end = False
        self.completed = False
        self.mask_dir = mask_dir

        self.out_w = self.video_info.width
        self.out_h = self.video_info.height
        if output_height > 0:
            self.out_w = int(output_height * 2)
            self.out_h = int(output_height)
            print("use custom output resolution", f"{self.out_w}:{self.out_h}")
        
        self.config = {"parameter": {"width": self.out_w, "height": self.out_h}}
        if eq2fisheye:
            self.config["filter_complex"] = (
                f"[0:v]split=2[left][right]; "
                f"[left]crop=ih:ih:0:0[left_crop]; "
                f"[right]crop=ih:ih:ih:0[right_crop]; "
                f"[left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; "
                f"[right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; "
                f"[leftfisheye][rightfisheye]hstack,scale={self.out_w}:{self.out_h}[v]"
            )
        else:
            self.config["filter_complex"] = f"[0:v]scale={self.out_w}:{self.out_h}[v]"

        self.ffmpeg = FFmpegStream(video_path=video, config=self.config, skip_frames=0)
        base, ext = os.path.splitext(os.path.basename(video))
        self.out_path = f"{base}-merged{ext}"
        self.result_name = f"{base}-alpha{ext}"
        self.writer = ArVideoWriter(self.out_path, self.video_info.fps, crf=crf)
        
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        i = 0
        while not self.end:
            while self.queue.qsize() == 0:
                time.sleep(1)
            start = i
            end = self.queue.get()
            if end < 0:
                print("End marker detected")
                self.end = True
                break
            if start > end:
                print("Error frame already processed", start)
            else:
                for _ in range(start, end):
                    frame = self.ffmpeg.read()
                    if frame is None:
                        print("Video does not have more frames")
                        self.end = True
                        break

                    print("process frame", i)
                    mask_path = os.path.join(self.mask_dir, f"{i+1:06d}.png")
                    if not os.path.exists(mask_path):
                        print("No more mask files")
                        self.end = True
                        break

                    alpha = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    self.writer.add_frame(frame, alpha)
                    i += 1

        self.ffmpeg.stop()
        self.writer.finalize()
        while not self.writer.is_finished():
            time.sleep(0.1)
        print("AR writer completed with", i, "frames")

        subprocess.run([
            "ffmpeg",
            "-i", self.out_path,
            "-i", self.video_path,
            "-c", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            self.result_name
        ])

        os.remove(self.out_path)
        print("Merged with audio")
        self.completed = True

    def get_video_path(self):
        if not self.is_finished():
            return None
        return self.result_name

    def is_finished(self):
        return self.completed

    def set_batch(self, frame_end):
        while self.queue.full():
            time.sleep(1)
        self.queue.put(frame_end)

    def set_end(self):
        while self.queue.full():
            time.sleep(1)
        self.queue.put(-1)


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

    ar_writer = ArVideoWriterWrapper(args.video, args.mask_dir, args.output_height, args.eq2fisheye, args.crf)
    ar_writer.set_batch(args.batch)
    ar_writer.set_end()
    while not ar_writer.is_finished():
        time.sleep(1)

    print("completed", ar_writer.get_video_path())
