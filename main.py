import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import traceback
import gradio as gr
import subprocess
import tempfile
import shutil
import pickle
import torch
import cv2
import gc
import glob
import time
import threading
import queue
import urllib3
import asyncio
import subprocess
import requests
import random
from sam2_executor import GroundingDinoSAM2Segment, SAM2PointSegment
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

from data.ffmpegstream import FFmpegStream
from data.ArVideoWriter import ArVideoWriter
from video_process import ImageFrame
from filebrowser_client import FilebrowserClient

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

WORKER_STATUS = "Idle"
MASK_SIZE = 1440
SECONDS = 10
WARMUP = 4
JOB_VERSION = 4
SSIM_THRESHOLD = float(os.environ.get('SSIM_THRESHOLD', "0.983"))
DEBUG = False
MASK_DEBUG = os.environ.get('MASK_DEBUG', 'False').lower() == 'true'
SURPLUS_IGNORE = os.environ.get('SURPLUS_IGNORE', 'True').lower() == 'true'
SCHEDULE = os.environ.get('EXECUTE_SCHEDULER_ON_START', 'True').lower() == 'true'


DEVICE_JOB = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_choices = []
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / (1024**3)
        gpu_choices.append(f"GPU {i}: {props.name} ({total_vram_gb:.1f} GB)")
else:
    gpu_choices = ["CPU"]

def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)

def prepare_frame(frame):
    global DEVICE_JOB
    vframes = torch.from_numpy(frame)
        
    if vframes.shape[-1] == 3:
        vframes = vframes.permute(2, 0, 1)
    
    vframes =  vframes.to(DEVICE_JOB)

    image_input = vframes.float() / 255.0

    return image_input

def prepare_mask(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)

    return mask

def finalize_mask(mask):
    global DEVICE_JOB
    mask = torch.from_numpy(mask)
    mask = mask.to(DEVICE_JOB)

    return mask

def fix_mask2(mask):
    global DEVICE_JOB
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask)
    mask = mask.to(DEVICE_JOB)

    return mask


@torch.no_grad()
def process_with_reverse_tracking(video, projection, masks, crf = 16, erode = False, force_init_mask=False, output_height=0, keepEq=False):
    global WORKER_STATUS
    global DEVICE_JOB

    maskIdx = 0
    mask_w, mask_h = masks[maskIdx]['maskL'].size

    original_filename = os.path.basename(video)
    file_name, file_extension = os.path.splitext(original_filename)
    
    video_info = FFmpegStream.get_video_info(video)
    
    reader_config = {
        "parameter": {
            "width": 2*mask_w,
            "height": mask_h,
        }
    }
    
    output_w = 2*mask_w
    if not keepEq and "eq" == projection:
        reader_config["filter_complex"] = f"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack,scale={output_w}:{mask_h}[v]"
        projection_out = "fisheye180"
    else:
        projection_out = projection
        reader_config["video_filter"] = f"scale={output_w}:{mask_h}"

    WORKER_STATUS = f"Load Models to create Masks"

    current_frame = 0
    objects = [1]

    ffmpeg = FFmpegStream(
        video_path = video,
        config = reader_config,
        skip_frames = 0,
        watchdog_timeout_in_seconds = 0 # we can not use wd here
    )

    result_name = file_name + "_" + str(projection_out).upper() + "_alpha" + file_extension

    matanyone1 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
    processor1 = InferenceCore(matanyone1, cfg=matanyone1.cfg, device=DEVICE_JOB)

    matanyone2 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
    processor2 = InferenceCore(matanyone2, cfg=matanyone2.cfg, device=DEVICE_JOB)

    for i in range(len(masks)):
        imgLV = prepare_frame(masks[i]['frameL'])
        imgRV = prepare_frame(masks[i]['frameR'])
        imgLMask = fix_mask2(masks[i]['maskL'])
        imgRMask = fix_mask2(masks[i]['maskR'])
        _ = processor1.step(imgLV, imgLMask, objects=objects, force_permanent=True)
        _ = processor2.step(imgRV, imgRMask, objects=objects, force_permanent=True)

    if os.path.exists("process/frames"):
        shutil.rmtree("process/frames")

    if os.path.exists("process/masks"):
        shutil.rmtree("process/masks")

    if os.path.exists("process/debug"):
        shutil.rmtree("process/debug")

    os.makedirs("process", exist_ok=True)
    os.makedirs("process/frames", exist_ok=True)
    os.makedirs("process/masks", exist_ok=True)
    os.makedirs("process/debug", exist_ok=True)
    reverse_track = False

    WORKER_STATUS = "Process Video..."
    while ffmpeg.isOpen():
        img = ffmpeg.read()
        if img is None:
            break
        current_frame += 1

        img_scaled = img

        _, width = img_scaled.shape[:2]
        imgL = img_scaled[:, :int(width/2)]
        imgR = img_scaled[:, int(width/2):]

        frame_match = False
        if force_init_mask and current_frame == 1:
            frame_match = True

        if maskIdx < len(masks):
            s1 = ssim(masks[maskIdx]['frameLGray'], cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
            if DEBUG:
                print("ssim1", s1)
            if s1 > SSIM_THRESHOLD:
                s2 = ssim(masks[maskIdx]['frameRGray'], cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))
                if not DEBUG:
                    print("ssim1", s1)
                print("ssim2", s2)
                if s2 > SSIM_THRESHOLD:
                    frame_match = True

        imgLV = prepare_frame(imgL)
        imgRV = prepare_frame(imgR)

        if frame_match:
            print("match at", current_frame)
            mL = masks[maskIdx]['maskL']
            mR = masks[maskIdx]['maskR']
            imgLMask = fix_mask2(mL)
            imgRMask = fix_mask2(mR)
            maskIdx += 1
            reverse_track = True

            output_prob_L = processor1.step(imgLV, imgLMask, objects=objects)
            output_prob_R = processor2.step(imgRV, imgRMask, objects=objects)
 
            for _ in range(WARMUP):
                output_prob_L = processor1.step(imgLV, first_frame_pred=maskIdx==1)
                output_prob_R = processor2.step(imgRV, first_frame_pred=maskIdx==1)

            if MASK_DEBUG:
                combined_mask = cv2.hconcat([np.array(mL), np.array(mR)])
                mask = Image.fromarray(combined_mask).convert("L")
                preview = Image.composite(
                    Image.new("RGB", (combined_mask.shape[1], combined_mask.shape[0]), "blue"),
                    Image.fromarray(img_scaled).convert("RGBA"),
                    mask.point(lambda p: 100 if p > 1 else 0)
                )
                preview.save('process/debug/match_' + str(current_frame).zfill(6) + ".png")

        elif maskIdx > 0:
            output_prob_L = processor1.step(imgLV)
            output_prob_R = processor2.step(imgRV)
        else:
            print("Warning: Start frame not found yet")
            cv2.imwrite('process/frames/' + str(current_frame).zfill(6) + ".png", img_scaled)
            combined_mask = np.zeros((img_scaled.shape[0], img_scaled.shape[1]), dtype=np.uint8)
            cv2.imwrite('process/masks/' + str(current_frame).zfill(6) + ".png", combined_mask)
            continue

        WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length}"

        mask_output_L = processor1.output_prob_to_mask(output_prob_L)
        mask_output_R = processor2.output_prob_to_mask(output_prob_R)

        mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
        mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

        mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
        mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

        combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])

        if maskIdx < len(masks):
            cv2.imwrite('process/frames/' + str(current_frame).zfill(6) + ".png", img_scaled)
        
        cv2.imwrite('process/masks/' + str(current_frame).zfill(6) + ".png", combined_mask)
        if MASK_DEBUG:
            mask = Image.fromarray(combined_mask).convert("L")
            preview = Image.composite(
                Image.new("RGB", (combined_mask.shape[1], combined_mask.shape[0]), "blue"), 
                Image.fromarray(img_scaled).convert("RGBA"),
                mask.point(lambda p: 100 if p > 1 else 0)
            )
            preview.save('process/debug/forward_' + str(current_frame).zfill(6) + ".png")

        if reverse_track:
            imgLV_end = imgLV
            imgRV_end = imgRV
            reverse_track = False
            frame_files = sorted(['process/frames/' + f for f in os.listdir('process/frames/') if f.endswith(".png")],  reverse=True)
            subprocess_len = len(frame_files)
            for idx, frame_file in enumerate(frame_files):
                img_scaled = cv2.imread(frame_file)
                os.remove(frame_file)

                _, width = img_scaled.shape[:2]
                imgL = img_scaled[:, :int(width/2)]
                imgR = img_scaled[:, int(width/2):]

                imgLV = prepare_frame(imgL)
                imgRV = prepare_frame(imgR)

                output_prob_L = processor1.step(imgLV)
                output_prob_R = processor2.step(imgRV)
                WORKER_STATUS = f"Create Mask {current_frame}/{video_info.length} - Subprocess {idx}/{subprocess_len}"

                mask_output_L = processor1.output_prob_to_mask(output_prob_L)
                mask_output_R = processor2.output_prob_to_mask(output_prob_R)

                mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
                mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

                mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
                mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

                combined_mask = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
                maskA = cv2.imread(frame_file.replace('frames', 'masks'), cv2.IMREAD_UNCHANGED)
                
                if MASK_DEBUG:
                    # cv2.imwrite(frame_file.replace('frames', 'debug').replace('.png', '_rev.png'), combined_mask)
                    mask = Image.fromarray(combined_mask).convert("L")
                    preview = Image.composite(
                        Image.new("RGB", (combined_mask.shape[1], combined_mask.shape[0]), "blue"),
                        Image.fromarray(img_scaled).convert("RGBA"),
                        mask.point(lambda p: 100 if p > 1 else 0)
                    )
                    preview.save(frame_file.replace('frames', 'debug').replace('.png', '_rev.png'))

                
                # using avg or other merge gives much worse reults at edges
                mergedA = np.bitwise_or(np.array(maskA), np.array(combined_mask))

                cv2.imwrite(frame_file.replace('frames', 'masks'), mergedA)
                if MASK_DEBUG:
                    mask = Image.fromarray(mergedA).convert("L")
                    preview = Image.composite(
                        Image.new("RGB", (mergedA.shape[1], mergedA.shape[0]), "blue"),
                        Image.fromarray(img_scaled).convert("RGBA"),
                        mask.point(lambda p: 100 if p > 1 else 0)
                    )
                    preview.save(frame_file.replace('frames', 'debug').replace('.png', '_res.png'))

            print("reverse tracking of", subprocess_len, "completed")
            # set model state to forware tracking again
            imgLMask = fix_mask2(masks[maskIdx-1]['maskL'])
            imgRMask = fix_mask2(masks[maskIdx-1]['maskR'])
            output_prob_L = processor1.step(imgLV_end, imgLMask, objects=objects)
            output_prob_R = processor2.step(imgRV_end, imgRMask, objects=objects)
            for _ in range(WARMUP):
                output_prob_L = processor1.step(imgLV_end)
                output_prob_R = processor2.step(imgRV_end)
        
        gc.collect()

    shutil.rmtree("process/frames")
    if maskIdx < len(masks):
        print("ERROR: not all frames found in video!")

    del processor1
    del processor2
    del matanyone1
    del matanyone2

    if torch.torch.cuda.is_available():
        torch.cuda.empty_cache()

    ffmpeg.stop()

    gc.collect()

    WORKER_STATUS = f"Create Mask Video..."
    print("create Video", result_name)


    out_resolution = f'{video_info.width}:{video_info.height}'
    scale = video_info.height / mask_h * 0.4
    if output_height > 0:
        out_w = int(output_height * 2)
        out_h = int(output_height)
        scale = out_h / mask_h * 0.4
        out_resolution = f'{out_w}:{out_h}'
        print("use custom output resolution", out_resolution)

    fc2 = f'"[1]scale=iw*{scale}:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h"'


    cmd = [
        "ffmpeg",
        '-hide_banner',
        '-loglevel', 'warning',
        '-thread_queue_size', '64',
        '-ss', FFmpegStream.frame_to_timestamp(0, video_info.fps),
        '-hwaccel', 'auto',
        '-i', "\""+str(video)+"\"",
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vsync', 'passthrough',
        '-vcodec', 'rawvideo',
        '-an',
        '-sn'
    ]

    if not keepEq and "eq" == projection:
        cmd += [
            "-filter_complex", f"\"[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack,scale={out_resolution}[v]\"",
            "-map", "[v]"
        ]
    else:
        cmd += [
            "-filter_complex", f"\"[0:v]scale={out_resolution}[v]\"",
            "-map", "[v]"
        ]

    cmd += [
        '-',
        '|',
        "ffmpeg",
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', out_resolution,
        '-r', str(video_info.fps),
        '-thread_queue_size', '64',
        '-i', 'pipe:0',
        '-r', str(video_info.fps),
        '-thread_queue_size', '64',
        "-i", "\"process/masks/%06d.png\"",
        "-i", "mask.png",
        '-r', str(video_info.fps),
        '-i', "\""+str(video)+"\"",
        "-filter_complex",
        fc2,
        "-c:v", "libx265", 
        "-crf", str(crf),
        "-preset", "veryfast",
        "-map", "\"3:a:?\"",
        "-c:a", "copy",
        "\""+result_name+"\"",
        "-y"
    ]

    if DEBUG:
        print(cmd)

    subprocess.run(' '.join(cmd), shell=True)

    shutil.rmtree("process/masks")

    WORKER_STATUS = f"Convertion completed" 
    return result_name

result_list = []
frame_name = None

def background_worker():
    global WORKER_STATUS
    surplus_url = os.environ.get('JOB_SURPLUS_CHECK_URL')
    while True:
        if not SCHEDULE:
            time.sleep(3)
            continue

        if surplus_url and not SURPLUS_IGNORE:
            try:
                start_job = "True" in str(requests.get(surplus_url, verify=False).json())
            except Exception as ex:
                start_job = False
                # print(ex)

            if not start_job:
                WORKER_STATUS = "Wait for surplus"
                time.sleep(30)
                continue

        pkl = [x for x in glob.glob("/jobs/*.pkl")]

        if len(pkl) == 0:
            time.sleep(2)
            continue

        pkl = sorted(pkl)[0]
        time.sleep(2) # ensure file is fully written
        with open(pkl, 'rb') as f:
            job = pickle.load(f)

        if job['version'] == JOB_VERSION:
            print("Start job", pkl)
            try:
                result = process_with_reverse_tracking(job['video'], job['projection'], job['masks'], job['crf'], job['erode'], job['forceInitMask'], job['outputHeight'], job['keepEq'])
                
                if result is not None and os.path.exists(result):
                    result_list.append(result)
                    if filebrowser_host := os.environ.get('FILEBROWSER_HOST'):
                        WORKER_STATUS = "Uploading..."
                        try:
                            if filebrowser_user := os.environ.get('FILEBROWSER_USER'):
                                client = FilebrowserClient(filebrowser_host, username=filebrowser_user, password=os.environ.get('FILEBROWSER_PASSWORD'), insecure=True)
                            else:
                                client = FilebrowserClient(filebrowser_host, insecure=True)

                            asyncio.run(client.connect())
                            couroutine = client.upload(
                                local_path=result,
                                remote_path=os.environ.get('FILEBROWSER_PATH', os.path.basename(result)).replace("{filename}", os.path.basename(result)),
                                override=True,
                                concurrent=1,
                            )
                            asyncio.run(couroutine)
                        except Exception as ex:
                            print("upload failed", ex)
            except:
                traceback.print_exc()


        os.makedirs('/jobs/completed', exist_ok=True)
        shutil.move(job['video'], os.path.join('/jobs/completed', os.path.basename(job['video'])))
        shutil.move(pkl, os.path.join('/jobs/completed', os.path.basename(pkl)))
        print('job', pkl, 'completed')

        time.sleep(1)
        WORKER_STATUS = "Idle"

def add_job(video, projection, crf, erode, forceInitMask, video_output_height, keep_eq):
    RETURN_VALUES = 16
    if video is None:
        gr.Warning("Could not add Job: Video not found", duration=5)
        return tuple(gr.skip() for _ in range(RETURN_VALUES))

    masksFiles = sorted([f for f in os.listdir('masksL') if f.endswith(".png") and os.path.exists(os.path.join('masksR', f))])
    masks = []
    for f in masksFiles:
        frame = cv2.imread(os.path.join('frames', f))
        width = 2*MASK_SIZE

        frameL = frame[:, :int(width/2)]
        frameR = frame[:, int(width/2):]

        maskL = Image.open(os.path.join('masksL', f)).convert('L')
        maskR = Image.open(os.path.join('masksR', f)).convert('L')

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        masks.append({
            'maskL': maskL,
            'maskR': maskR,
            'frameL': frameL,
            'frameR': frameR,
            'frameLGray': grayL,
            'frameRGray': grayR
        })

    if len(masks) == 0:
        gr.Warning("Could not add Job: Mask Missing", duration=5)
        return tuple(gr.skip() for _ in range(RETURN_VALUES))

    ts = str(int(time.time()))

    dest = '/jobs/' + ts + "_" + os.path.basename(video.name)
    shutil.move(video.name, dest)

    job_data = {
        'version': JOB_VERSION,
        'video': dest,
        'projection': projection,
        'crf': crf,
        'masks': masks,
        'erode': erode,
        'forceInitMask': forceInitMask,
        'outputHeight':  video_output_height,
        'keepEq': keep_eq
    }

    with open(f"/jobs/{ts}.pkl", "wb") as f:
        pickle.dump(job_data, f)

    return tuple(None for _ in range(RETURN_VALUES))

def status_text():
    pending_jobs = len([x for x in glob.glob("/jobs/*.pkl")])
    return "Worker Status: " + WORKER_STATUS + "\n" \
        + f"Pending Jobs: {pending_jobs}"

current_origin_frame = {
    "L": None,
    "R": None
}

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

def add_mark(frame):
    if frame is None:
        return None
    result = frame.frame_data.copy()
    marker_size = 25
    marker_thickness = 3
    marker_default_width = 1200
    width = result.shape[0]
    ratio = width / marker_default_width
    marker_final_size = int(marker_size * ratio)
    if marker_final_size < 3:
        marker_final_size = 3
    marker_final_thickness = int(marker_thickness * ratio)
    if marker_final_thickness < 2:
        marker_final_thickness = 2
    for (x, y, label) in frame.point_set:
        cv2.drawMarker(result, (x, y), colors[label], markerType=markers[label], markerSize=marker_final_size, thickness=marker_final_thickness)
    return result


def generate_gallery_list():
    frames = sorted([os.path.join('frames', f) for f in os.listdir('frames') if f.endswith(".png")])
    for idx in range(len(frames)):
        if os.path.exists(frames[idx].replace('frames', 'previews')):
            frames[idx] = frames[idx].replace('frames', 'previews')
    gallery_items = [(frame, str(str(max(0, int(idx*SECONDS - SECONDS/2))) + " sec")) for idx, frame in enumerate(frames)]
    return gallery_items

def set_mask_size(x):
    global MASK_SIZE
    MASK_SIZE = int(x)
    print("set mask size to", MASK_SIZE)

def set_extract_frames_step(x):
    global SECONDS
    SECONDS = int(x)
    print("set extract seconds to", SECONDS)

def set_prefered_mask_size(input_video):
    global MASK_SIZE
    try:
        video_info = FFmpegStream.get_video_info(input_video)
        prefered_mask_height = round(video_info.height * 0.4 / 2.0) * 2
        set_mask_size(max((820, prefered_mask_height)))
    except Exception as ex:
        print(ex)
    
    return MASK_SIZE

def get_prevered_output_height(height):
    for h in [2048, 3072, 3600, 3840, 4000, 4096]:
        if h >= height:
            return h

    return 0

def extract_frames(video, projection, mask_size, frames_seconds, keep_eq):
    set_mask_size(mask_size)
    set_extract_frames_step(frames_seconds)
    for dir in ["frames", "previews", "masksL", 'masksR']:
        if os.path.exists(dir):
            shutil.rmtree(dir)

        os.makedirs(dir, exist_ok=True)

    if not keep_eq and str(projection) == "eq":
        filter_complex = "split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]"
        os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -frames:v 1 -filter_complex \"[0:v]{filter_complex}\" -map \"[v]\" frames/0000.png")
        if SECONDS > 0:
            os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -filter_complex \"[0:v]fps=1/{SECONDS},{filter_complex}\" -map \"[v]\" -start_number 1 frames/%04d.png")
    else:
        os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -frames:v 1 -pix_fmt bgr24 frames/0000.png")
        if SECONDS > 0:
            os.system(f"ffmpeg -hide_banner -loglevel warning -hwaccel auto -i \"{video.name}\" -vf fps=1/{SECONDS} -pix_fmt bgr24 -start_number 1 frames/%04d.png")
    
    frames = [os.path.join('frames', f) for f in os.listdir('frames') if f.endswith(".png")]
    out_height = 0

    #NOTE: use same method for resizing to get pixel exact matches
    for frame in frames:
        frame_data = cv2.imread(frame)
        out_height = frame_data.shape[0]
        img_scaled = cv2.resize(frame_data, (2*MASK_SIZE, MASK_SIZE))
        cv2.imwrite(frame, img_scaled)

    return generate_gallery_list(), get_prevered_output_height(out_height)

def get_selected(selected):
    global frame_name
    if selected is None or "image" not in selected:
        return None, None, None, None, None, None

    frame_name = os.path.basename(selected['image']['path'])
    if os.path.exists(os.path.join('frames', frame_name)):
        frame = cv2.imread(os.path.join('frames', frame_name))
    else:
        return None, None, None, None, None, None
    
    width = 2*MASK_SIZE

    frameL = frame[:, :int(width/2)]
    frameR = frame[:, int(width/2):]

    frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)

    current_origin_frame['L'] = ImageFrame(frameL, 0)
    current_origin_frame['R'] = ImageFrame(frameR, 0)

    maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")
    maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    if os.path.exists(os.path.join('masksL', frame_name)):
        maskL = Image.open(os.path.join('masksL', frame_name)).convert("L")

    if os.path.exists(os.path.join('masksR', frame_name)):
        maskR = Image.open(os.path.join('masksR', frame_name)).convert("L")

    return Image.fromarray(frameL), \
        Image.fromarray(frameR), \
        Image.fromarray(frameL), \
        Image.fromarray(frameR), \
        maskL, \
        maskR

def get_mask(frameL, frameR, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold, maskLNegativePrompt, maskRNegativePrompt):
    if frameL is None or frameR is None:
        return None, None, None, None

    sam2 = GroundingDinoSAM2Segment()

    if len( maskLPrompt) > 0: 
        (_, imgLMask) = sam2.predict([frameL], maskLThreshold, maskLPrompt)
        maskL = (imgLMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
        if len( maskLNegativePrompt) > 0: 
            (_, negativeImgLMask) = sam2.predict([frameL], maskLThreshold, maskLNegativePrompt)
            invNegativeMaskL = 255 - (negativeImgLMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
            maskL = np.bitwise_and(maskL, invNegativeMaskL)

        maskL = Image.fromarray(maskL).convert('L')

        previewL = Image.composite(
            Image.new("RGB", maskL.size, "blue"),
            Image.fromarray(frameL).convert("RGBA"),
            maskL.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewL = frameL
        maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    if len( maskRPrompt) > 0: 
        (_, imgRMask) = sam2.predict([frameR], maskRThreshold, maskRPrompt)

        maskR = (imgRMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)

        if len( maskRNegativePrompt) > 0: 
            (_, negativeImgRMask) = sam2.predict([frameR], maskRThreshold, maskRNegativePrompt)
            invNegativeMaskR = 255 - (negativeImgRMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
            maskR = np.bitwise_and(maskR, invNegativeMaskR)

        maskR = Image.fromarray(maskR).convert('L')
        previewR = Image.composite(
                Image.new("RGB", maskR.size, "blue"),
                Image.fromarray(frameR).convert("RGBA"),
                maskR.point(lambda p: 100 if p > 1 else 0)
            )
    else:
        previewR = frameR
        maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L")

    del sam2
    
    return previewL, maskL, previewR, maskR

def get_mask2():
    global current_origin_frame

    if current_origin_frame['L'] is None or current_origin_frame['R'] is None:
        return None, None, None, None

    frameL = current_origin_frame['L'].frame_data
    frameR = current_origin_frame['R'].frame_data

    pointsL = []
    for x, y, label in current_origin_frame['L'].point_set:
        pointsL.append(((x, y), label))

    pointsR = []
    for x, y, label in current_origin_frame['R'].point_set:
        pointsR.append(((x, y), label))

    sam2 = SAM2PointSegment()

    if len(pointsL) != 0:
        (_, imgLMask) = sam2.predict(frameL, pointsL)
        maskL = Image.fromarray((imgLMask * 255).astype(np.uint8)).convert('L')
        previewL = Image.composite(
            Image.new("RGB", maskL.size, "blue"),
            Image.fromarray(frameL).convert("RGBA"),
            maskL.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewL = frameL
        maskL = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L") 


    if len(pointsR) != 0:
        (_, imgRMask) = sam2.predict(frameR, pointsR)
    
        maskR = Image.fromarray((imgRMask * 255).astype(np.uint8)).convert('L')

        previewR = Image.composite(
            Image.new("RGB", maskR.size, "blue"),
            Image.fromarray(frameR).convert("RGBA"),
            maskR.point(lambda p: 100 if p > 1 else 0)
        )
    else:
        previewR = frameR
        maskR = Image.fromarray(np.zeros([MASK_SIZE, MASK_SIZE, 3], dtype=np.uint8)).convert("L") 

    del sam2

    return previewL, maskL, previewR, maskR

def generate_mask_preview(mask, eye: str):
    if mask is None:
        return None

    frame = current_origin_frame[eye].frame_data
    if frame is None:
        return None

    mask = Image.fromarray(mask).convert("L")
    preview = Image.composite(
        Image.new("RGB", mask.size, "blue"),
        Image.fromarray(frame).convert("RGBA"),
        mask.point(lambda p: 100 if p > 1 else 0)
    )

    return preview

def postprocess_mask(maskL, maskR, dilate, erode):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # first dilate then erode to ensure to fill holes
    maskL = cv2.dilate(maskL, kernel, iterations=dilate)
    maskL = cv2.erode(maskL, kernel, iterations=erode)
    maskR = cv2.dilate(maskR, kernel, iterations=dilate)
    maskR = cv2.erode(maskR, kernel, iterations=erode)

    maskL = Image.fromarray(maskL).convert("L")
    maskL.save(os.path.join('masksL', frame_name))
    maskR = Image.fromarray(maskR).convert("L")
    maskR.save(os.path.join('masksR', frame_name))

    pL = generate_mask_preview(np.array(maskL), 'L')
    pR = generate_mask_preview(np.array(maskR), 'R')
    if pL is not None and pR is not None:
        pL = cv2.cvtColor(np.array(pL), cv2.COLOR_RGB2BGR)
        pR = cv2.cvtColor(np.array(pR), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('previews', frame_name), cv2.hconcat([pL, pR]))

    return maskL, maskR, generate_gallery_list(), os.path.join('previews', frame_name)

def merge_add_mask(maskL, maskR, mergedMaskL, mergedMaskR, dilate, erode):
    if maskL is not None and mergedMaskL is not None:
        mergedMaskL = np.bitwise_or(np.array(mergedMaskL), np.array(maskL))
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        mergedMaskR = np.bitwise_or(np.array(mergedMaskR), np.array(maskR))
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")

    a, b, c, d = postprocess_mask(np.array(mergedMaskL), np.array(mergedMaskR), dilate, erode)

    return  mergedMaskL, mergedMaskR, a, b, c, d

def merge_subtract_mask(maskL, maskR, mergedMaskL, mergedMaskR, dilate, erode):
    if maskL is not None and mergedMaskL is not None:
        invMaskL = 255 - np.array(maskL)
        mergedMaskL = np.bitwise_and(np.array(mergedMaskL), invMaskL)
        mergedMaskL = Image.fromarray(mergedMaskL).convert("L")
    if maskR is not None and mergedMaskR is not None:
        invMaskR = 255 - np.array(maskR)
        mergedMaskR = np.bitwise_and(np.array(mergedMaskR), invMaskR)
        mergedMaskR = Image.fromarray(mergedMaskR).convert("L")

    a, b, c, d = postprocess_mask(np.array(mergedMaskL), np.array(mergedMaskR), dilate, erode) 

    return  mergedMaskL, mergedMaskR, a, b, c, d

def set_schedule(x):
    global SCHEDULE
    SCHEDULE = bool(x)

def set_sureplus_ignore(x):
    global SURPLUS_IGNORE
    SURPLUS_IGNORE = bool(x)

def generate_example(maskL, maskR):
    frameL = current_origin_frame['L'].frame_data
    frameR = current_origin_frame['R'].frame_data

    with torch.no_grad():
        matanyone1 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
        processor1 = InferenceCore(matanyone1, cfg=matanyone1.cfg, device=DEVICE_JOB)

        result = []

        maskL = np.array(Image.fromarray(maskL).convert("L"))
        maskR = np.array(Image.fromarray(maskR).convert("L"))
        for (frame, mask) in [(frameL, maskL), (frameR, maskR)]:
            objects = [1]
            mask = fix_mask2(mask)
            frame = prepare_frame(frame)
            output_prob = processor1.step(frame, mask, objects=objects)
            for _ in range(WARMUP):
                output_prob = processor1.step(frame, first_frame_pred=True)
            mask_output = processor1.output_prob_to_mask(output_prob)
            mask_output_pha = mask_output.unsqueeze(2).cpu().detach().numpy()
            mask_output_pha = (mask_output_pha*255).astype(np.uint8)
            result.append(mask_output_pha)

    # Workaround since Pillow can not laod directly?
    cv2.imwrite("tmp_l.png", result[0])
    cv2.imwrite("tmp_r.png", result[1])

    return generate_mask_preview(cv2.imread("tmp_l.png"), 'L'), generate_mask_preview(cv2.imread('tmp_r.png'), 'R')

def clear_completed_jobs():
    folder_path = '/jobs/completed'
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def on_gpu_change(selected_gpu):
    global DEVICE_JOB
    if "CPU" in selected_gpu:
        device = "cpu"
    else:
        gpu_index = int(selected_gpu.split(":")[0].split()[-1])
        device = f"cuda:{gpu_index}"
        DEVICE_JOB = torch.device(f"cuda:{gpu_index}")

    print("device", device)

with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    gr.Markdown('''
        Process:
        1. Upload Your video
        2. Select Video Source Format
        3. Select Mask Size (Higher value require more VRAM!)
        3. Extract Frames
        4. Generate Initial Mask. Use Points or Prompts to generate individual Masks and merge them with add or subtract button to the initial mask. To create a second mask with points you have to use the crea button to select the points for the next partial mask. Use the foreground and subtract to remove areas from mask. use foreground and add to add areas to the inital mask. You can also try to specify additional backrground points but for me this always results in worse results.
        5. Add Video Mask Job
    ''')
    with gr.Column():
        gr.Markdown("## Settings")
        gpu_dropdown = gr.Dropdown(choices=gpu_choices, label="Select GPU (currently only global for all jobs)", value=gpu_choices[0])
        gpu_dropdown.change(
            fn=on_gpu_change, 
            inputs=gpu_dropdown, 
            outputs=None
        )
        gr.Markdown("## Stage 1 - Video")
        input_video = gr.File(label="Upload Video (MKV or MP4)", file_types=["mkv", "mp4", "video"])
        gr.Markdown("## Stage 2 - Video Parameter")
        projection_dropdown = gr.Dropdown(choices=["eq", "fisheye180", "fisheye190", "fisheye200"], label="VR Video Source Format", value="eq")
        keep_eq = gr.Checkbox(label="Keep Equirectangular Format. Do not convert to fisheye view. (HereSphere VR Player can play equirectangular with packed alpha, but some artifacts appear at the 180° boundary)", value=False, info="")
        mask_size = gr.Number(
            label="Mask Size is set automatically, but you can change it manually e.g. if your comupter have not enough VRAM (Mask Size larger than 40% of video height make no sense, higher mask value require more VRAM, 1440 require up to 20GB VRAM)",
            minimum=512,
            maximum=2048,
            step=1,
            value=MASK_SIZE
        )
        extract_frames_step = gr.Number(
            label="Extract frames every x seconds",
            minimum=0,
            maximum=600,
            step=1,
            value=SECONDS
        )
        input_video.change(
            fn=set_prefered_mask_size,
            inputs=input_video,
            outputs=mask_size,
        )

    with gr.Column():
        gr.Markdown("## Stage 3 - Extract First Frame")
        frame_button = gr.Button("Extract Projection Frame")
        gr.Markdown("### Frames")
        gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, object_fit="contain")
        select_button = gr.Button("Load Slected Projection Frame")
        with gr.Row():
            framePreviewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            framePreviewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

    gr.Markdown("## Stage 4 - Generate Mask")
    with gr.Tabs():
        with gr.Tab("Prompt"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        maskLPrompt = gr.Textbox(label="Positive Prompt", value="top person", lines=1, max_lines=1)
                        maskLNegativePrompt = gr.Textbox(label="Negative Prompt", value="", lines=1, max_lines=1)
                        maskLThreshold = gr.Number(
                            label="Threshold",
                            minimum=0.01,
                            maximum=0.99,
                            step=0.01,
                            value=0.3
                        )
                    with gr.Column():
                        maskRPrompt = gr.Textbox(label="Positive Prompt", value="top person", lines=1, max_lines=1)
                        maskRNegativePrompt = gr.Textbox(label="Negative Prompt", value="", lines=1, max_lines=1)
                        maskRThreshold = gr.Number(
                            label="Threshold",
                            minimum=0.01,
                            maximum=0.99,
                            step=0.01,
                            value=0.3
                        )
                mask_button = gr.Button("Generate Mask")
        with gr.Tab("Points"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        undoL = gr.Button('Undo')
                        removeL = gr.Button('Clear')
                    with gr.Column():
                        maskSelectionL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
                        groupL = gr.Radio(['foreground', 'background'], label='Object Type', value='foreground')
                with gr.Column():
                    with gr.Row():
                        undoR = gr.Button('Undo')
                        removeR = gr.Button('Clear')
                    with gr.Column():
                        maskSelectionR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
                        groupR = gr.Radio(['foreground', 'background'], label='Object Type', value='foreground')

            def add_mark_point(t, point_type, event: gr.SelectData):
                global current_origin_frame
                label = 1
                if point_type == 'foreground':
                    label = 1
                elif point_type == 'background':
                    label = 0
                
                if current_origin_frame[t] is None:
                    return None
                
                current_origin_frame[t].add(*event.index, label)
                return add_mark(current_origin_frame[t])

            def add_mark_point_l(point_type, event: gr.SelectData):
                return add_mark_point('L', point_type, event)
            
            def add_mark_point_r(point_type, event: gr.SelectData):
                return add_mark_point('R', point_type, event)
            
            def undo_last_point_l():
                global current_origin_frame
                if current_origin_frame['L'] is None:
                    return None
                current_origin_frame['L'].pop()
                return add_mark(current_origin_frame['L'])

            def undo_last_point_r():
                global current_origin_frame
                if current_origin_frame['R'] is None:
                    return None
                current_origin_frame['R'].pop()
                return add_mark(current_origin_frame['R'])

            def remove_all_points_l():
                global current_origin_frame
                if current_origin_frame['L'] is None:
                    return None
                current_origin_frame['L'].clear()
                return current_origin_frame['L'].frame_data
            
            def remove_all_points_r():
                global current_origin_frame
                if current_origin_frame['R'] is None:
                    return None
                current_origin_frame['R'].clear()
                return current_origin_frame['R'].frame_data

            gr.Image.select(maskSelectionL, add_mark_point_l, inputs = [groupL], outputs = [maskSelectionL])
            gr.Image.select(maskSelectionR, add_mark_point_r, inputs = [groupR], outputs = [maskSelectionR])

            gr.Button.click(undoL, undo_last_point_l, inputs=None, outputs=maskSelectionL)
            gr.Button.click(undoR, undo_last_point_r, inputs=None, outputs=maskSelectionR)
            gr.Button.click(removeL, remove_all_points_l, inputs=None, outputs=maskSelectionL)
            gr.Button.click(removeR, remove_all_points_r, inputs=None, outputs=maskSelectionR)
            mask2_button = gr.Button("Generate Mask")


    with gr.Column():
        gr.Markdown("### Mask Step 1")
        gr.Markdown("Preview")
        with gr.Row():
            maskPreviewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            maskPreviewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
        gr.Markdown("Mask")
        with gr.Row():
            maskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            maskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
    with gr.Column():
        gr.Markdown("### Mask Step 2")
        gr.Markdown("Add or subtract mask from mask step 1 to the frame initial Mask")
        with gr.Row():
            mask_add_button = gr.Button("Add Mask")
            mask_subtract_button = gr.Button("Subtract Mask")

        gr.Markdown("Combinend Mask")
        with gr.Row():
            mergedMaskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            mergedMaskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Postporcessed Mask")
        mask_dilate = gr.Slider(minimum=0, maximum=10, step=1, value=1, label="Dilate Iterrations")
        mask_erode = gr.Slider(minimum=0, maximum=10, step=1, value=3, label="Erode Iterrations")
        with gr.Row():
            postprocessedMaskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            postprocessedMaskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Preview")
        previewMergedMask = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        gr.Markdown("Optional: Generate MatAnyone Example Output (For debug purpose only)")
        example_button = gr.Button("Generate Example Output")
        with gr.Row():
            exampleL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
            exampleR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

        mask_dilate.change(
            fn=postprocess_mask,
            inputs=[mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        mask_erode.change(
            fn=postprocess_mask,
            inputs=[mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        selected_frame = gr.State(None)
        def store_index(evt: gr.SelectData):
            return evt.value

        gallery.select(store_index, None, selected_frame)
        select_button.click(
            fn=get_selected,
            inputs=[selected_frame],
            outputs=[framePreviewL, framePreviewR, maskSelectionL, maskSelectionR, mergedMaskL, mergedMaskR]
        )

        mask_button.click(
            fn=get_mask,
            inputs=[framePreviewL, framePreviewR, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold, maskLNegativePrompt, maskRNegativePrompt],
            outputs=[maskPreviewL, maskL, maskPreviewR, maskR]
        )

        mask2_button.click(
            fn=get_mask2,
            inputs=None,
            outputs=[maskPreviewL, maskL, maskPreviewR, maskR]
        )

        mask_add_button.click(
            fn=merge_add_mask,
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[mergedMaskL, mergedMaskR, postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        mask_subtract_button.click(
            fn=merge_subtract_mask,
            inputs=[maskL, maskR, mergedMaskL, mergedMaskR, mask_dilate, mask_erode],
            outputs=[mergedMaskL, mergedMaskR, postprocessedMaskL, postprocessedMaskR, gallery, previewMergedMask]
        )

        example_button.click(
            fn=generate_example,
            inputs=[postprocessedMaskL, postprocessedMaskR],
            outputs=[exampleL, exampleR]
        )

    with gr.Column():
        gr.Markdown("## Stage 5 - Add Job")
        crf_dropdown = gr.Dropdown(choices=[16,17,18,19,20,21,22], label="Encode CRF", value=16)
        erode_checkbox = gr.Checkbox(label="Erode Mask Output", value=True, info="")
        force_init_mask_checkbox = gr.Checkbox(label="Force Init Mask (Not recommend!)", value=False, info="")
        output_resolution_height = gr.Number(
            label="Set video output resolution height. (In HereSphere VR and DeoVR some resultion e.g. 1700 cause a invalid slightly shifted preview mask). Use a workign output video resolution height (2048, 3072, 3600, 3840, 4000, 4096). To keep original resolution set to 0",
            minimum=0,
            maximum=10000000,
            step=1,
            value=4096
        )
        add_button = gr.Button("Add Job")
        add_button.click(
            fn=add_job,
            inputs=[input_video, projection_dropdown, crf_dropdown, erode_checkbox, force_init_mask_checkbox, output_resolution_height, keep_eq],
            outputs=[input_video, framePreviewL, framePreviewR, maskPreviewL, mergedMaskL, maskPreviewR, mergedMaskR, maskL, maskR, maskSelectionL, maskSelectionR, previewMergedMask, exampleL, exampleR, postprocessedMaskL, postprocessedMaskR]
        )

    with gr.Column():
        gr.Markdown("## Job Control")
        schedule_checkbox = gr.Checkbox(label="Enable Job Scheduling", value=SCHEDULE, info="")
        ignore_surplus_checkbox = gr.Checkbox(label="Ignore Surplus Scheduling", value=SURPLUS_IGNORE, info="")
        clear_completed_jobs_button = gr.Button("Clear completed Jobs")
        restart_button = gr.Button("CLEANUP AND RESTART".upper())

        restart_button.click(
            # dirty hack, we use k8s restart pod
            fn=lambda: os.system("pkill python"),
            inputs=[],
            outputs=[]
        )

        clear_completed_jobs_button.click(
            fn=clear_completed_jobs
        )

        schedule_checkbox.change(
            fn=set_schedule,
            inputs=schedule_checkbox
        )

        ignore_surplus_checkbox.change(
            fn=set_sureplus_ignore,
            inputs=ignore_surplus_checkbox
        )

    with gr.Column():
        gr.Markdown("## Job Results")
        status = gr.Textbox(label="Status", lines=2)
        output_videos = gr.File(value=[], label="Download AR Videos", visible=True)

    frame_button.click(
        fn=extract_frames,
        inputs=[input_video, projection_dropdown, mask_size, extract_frames_step, keep_eq],
        outputs=[gallery, output_resolution_height]
    )

    timer1 = gr.Timer(2, active=True)
    timer5 = gr.Timer(5, active=True)
    timer1.tick(status_text, outputs=status)
    timer5.tick(lambda: result_list, outputs=output_videos)
    demo.load(fn=status_text, outputs=status)
    demo.load(fn=lambda: result_list, outputs=output_videos)


if __name__ == "__main__":
    print("gradio version", gr.__version__)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
    print("exit")
