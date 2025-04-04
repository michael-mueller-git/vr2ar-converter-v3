import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
import subprocess
import tempfile
import shutil
import torch
import cv2
import gc
import time
import threading
import queue
import random
from sam2_executor import GroundingDinoSAM2Segment
from PIL import Image

import torch.nn.functional as F
import numpy as np

from matanyone.model.matanyone import MatAnyone
from matanyone.inference.inference_core import InferenceCore

from data.ffmpegstream import FFmpegStream

WORKER_STATUS = "Idle"

def exec_ffmpeg(cmd, total_frames):
    global WORKER_STATUS
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
    for line in process.stdout:
        print(line.strip())
        if "frame=" in line:
            parts = line.split()
            try:
                current_frame = int(parts[1])
                percent = round((current_frame / total_frames) * 100, 1)
                WORKER_STATUS = f"Converting Video via FFmpeg {percent}%" 
            except ValueError:
                pass
    
    process.wait()
    
    if process.returncode != 0:
        print("ERROR", cmd)
        return False

    return True

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

def fix_mask_old(img, mask):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1)[0]
    
    # Convert mask to numpy for processing
    mask_np = (mask.cpu().numpy() * 255.0).astype(np.float32)
    mask_np = gen_dilate(mask_np, 10, 10)
    mask_np = gen_erosion(mask_np, 10, 10)

    mask_tensor = torch.from_numpy(mask_np)
   
    if torch.torch.cuda.is_available():
        mask_tensor = mask_tensor.cuda()

    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor  # Keep as [H, W]
    elif mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor[0]  # Convert [1, H, W] to [H, W]

    img_h, img_w = img.shape[1], img.shape[2]
    mask_h, mask_w = mask_tensor.shape
    
    if mask_h != img_h or mask_w != img_w:
        # Add batch and channel dimensions for interpolation
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_tensor = F.interpolate(mask_tensor, size=(img_h, img_w), mode='nearest')
        mask_tensor = mask_tensor[0, 0]  # Remove batch and channel dimensions

    return mask_tensor

def prepare_frame(frame, has_cuda=True):
    vframes = torch.from_numpy(frame)
        
    # Input is [B, H, W, C], convert to [B, C, H, W]
    if vframes.shape[-1] == 3:  # If channels are last
        vframes = vframes.permute(2, 0, 1)
    
    if has_cuda:
         vframes =  vframes.cuda()

    image_input = vframes.float() / 255.0

    return image_input

def fix_mask2(mask):
    mask = np.array(mask)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = torch.from_numpy(mask)
    if torch.torch.cuda.is_available():
        mask = mask.cuda()

    return mask


@torch.no_grad()
def process(job_id, video, projection, maskL, maskR):
    global WORKER_STATUS

    maskL = fix_mask2(maskL)
    maskR = fix_mask2(maskR)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_filename = os.path.basename(video.name)
        file_name, file_extension = os.path.splitext(original_filename)

        file_name = str(job_id).zfill(4) + "_" + file_name
        
        temp_input_path = os.path.join(temp_dir, original_filename)
        shutil.copy(video.name, temp_input_path)

        cap = cv2.VideoCapture(temp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        output_filename = f"{file_name}-fisheye.{file_extension}"
        output_path = os.path.join(temp_dir, output_filename)

        if str(projection) == "eq":
            cmd = [
                "ffmpeg",
                "-i", temp_input_path,
                "-filter_complex",
                "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]",
                "-map", "[v]",
                "-c:a", "copy",
                "-crf", "16",
                output_path
            ]
            projection = "fisheye180"
            if not exec_ffmpeg(cmd, total_frames):
                return (None, None)
        else:
            output_path = temp_input_path

        WORKER_STATUS = f"Load Models to create Masks"

        cap = cv2.VideoCapture(output_path)

        mask_video = file_name + "-alpha.avi"
        out = cv2.VideoWriter(
            mask_video,
            cv2.VideoWriter_fourcc(*'MJPG'),
            cap.get(cv2.CAP_PROP_FPS), 
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            False
        )
        current_frame = 0

        objects = [1]

        # TODO: when playing in Heresphere, the mask lags behind the video by approx. 1 frame. I still have to investigate where the problem lies. Possibly when merging the mask with the video or a bug in heresphere or the model simply always outputs a delayed mask, maybe it doesn't deal so well with the changes in movement. Workaround for now is to encode the mask of the next frame into the current frame.
        INSERT_SHIFT = True

        # adopted from published repo
        WARMUP = 10
        has_cuda = torch.torch.cuda.is_available()

        config = {
            "video_filter": "scale=${width}:${height}",
            "parameter": {
                "width": 2048,
                "height": 1024
            }
        }

        ffmpeg = FFmpegStream(
            video_path = output_path,
            config = config,
            skip_frames = 0
        )

        video_info = FFmpegStream.get_video_info(output_path)

        matanyone1 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
        if torch.torch.cuda.is_available():
            matanyone1 = matanyone1.cuda()
        matanyone1 = matanyone1.eval()
        processor1 = InferenceCore(matanyone1, cfg=matanyone1.cfg)

        matanyone2 = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
        if torch.torch.cuda.is_available():
            matanyone2 = matanyone2.cuda()
        matanyone2 = matanyone2.eval()
        processor2 = InferenceCore(matanyone2, cfg=matanyone2.cfg)

        while ffmpeg.isOpen():
            img = ffmpeg.read()
            if img is None:
                break
            current_frame += 1

            _, width = img.shape[:2]
            imgL = img[:, :int(width/2)]
            imgR = img[:, int(width/2):]

            imgLV = prepare_frame(imgL, has_cuda)
            imgRV = prepare_frame(imgR, has_cuda)

            if current_frame == 1:
                imgLMask = maskL
                imgRMask = maskR

                output_prob_L = processor1.step(imgLV, imgLMask, objects=objects)
                output_prob_R = processor2.step(imgRV, imgRMask, objects=objects)
                
                for i in range(WARMUP):
                    WORKER_STATUS = f"Warmup MatAnyone {i+1}/{WARMUP}"
                    output_prob_L = processor1.step(imgLV, first_frame_pred=True)
                    output_prob_R = processor2.step(imgRV, first_frame_pred=True)
            else:
                output_prob_L = processor1.step(imgLV)
                output_prob_R = processor2.step(imgRV)

            WORKER_STATUS = f"Create Mask {current_frame}/{total_frames}"
            print(WORKER_STATUS)

            mask_output_L = processor1.output_prob_to_mask(output_prob_L)
            mask_output_R = processor2.output_prob_to_mask(output_prob_R)

            mask_output_L_pha = mask_output_L.unsqueeze(2).cpu().detach().numpy()
            mask_output_R_pha = mask_output_R.unsqueeze(2).cpu().detach().numpy()

            mask_output_L_pha = (mask_output_L_pha*255).astype(np.uint8)
            mask_output_R_pha = (mask_output_R_pha*255).astype(np.uint8)

            mask_output_L_pha = cv2.erode(mask_output_L_pha, (3,3), iterations=1)
            mask_output_R_pha = cv2.erode(mask_output_R_pha, (3,3), iterations=1)

            combined_image = cv2.hconcat([mask_output_L_pha, mask_output_R_pha])
            combined_image = cv2.resize(combined_image, (video_info.width, video_info.height))

            
            _, binary = cv2.threshold(combined_image, 127, 255, cv2.THRESH_BINARY)

            if INSERT_SHIFT and current_frame != 1:
                out.write(binary)
            gc.collect()

        if INSERT_SHIFT:
            out.write(binary)

        ffmpeg.stop()
        out.release()

        del processor1
        del processor2
        del matanyone1
        del matanyone2

        if torch.torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        print("FFmpeg Convert #2")

        result_name = file_name.replace(' ', '_') + "_" + str(projection).upper() + "_alpha" + file_extension 
        cmd = [
            "ffmpeg",
            "-i", output_path,
            "-i", mask_video,
            "-i", "mask.png",
            "-i", temp_input_path,
            "-filter_complex",
            "[1]scale=iw*0.4:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h",
            "-c:v", "libx265", 
            "-crf", "16",
            "-preset", "veryfast",
            "-map", "3:a:?",
            "-c:a", "copy",
            result_name,
            "-y"
        ]

        if not exec_ffmpeg(cmd, total_frames):
            return (None, None)

    WORKER_STATUS = f"Convertion completed" 
    return (mask_video, result_name)


job_id = 0
task_queue = queue.Queue()
mask_list = []
result_list = []

def background_worker():
    global file_list
    global WORKER_STATUS
    while True:
        job = task_queue.get()
        if job is None:
            break
        print("Start job", job['id'])
        (mask, result) = process(job['id'], job['video'], job['projection'], job['maskL'], job['maskR'])
        if mask is not None:
            mask_list.append(mask)
        if result is not None:
            result_list.append(result)
        task_queue.task_done()
        time.sleep(5)
        WORKER_STATUS = "Idle"

def add_job(video, projection, maskL, maskR):
    global job_id
    if video is not None and maskL is not None and maskR is not None:
        job_id += 1
        print("Add job", job_id)
        task_queue.put({
            'id': job_id,
            'video': video,
            'projection': projection,
            'maskL': Image.fromarray(maskL).convert('L'),
            'maskR': Image.fromarray(maskR).convert('L')
        })
    return None, None, None, None, None

def status_text():
    global task_queue
    return "Worker Status: " + WORKER_STATUS + "\n" \
        + f"Pending Jobs: {task_queue.qsize()}"


def get_mask(video, projection, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold):
    frame = FFmpegStream.get_frame(video.name, 0)
    if str(projection) == "eq":
        configL = {
            "video_filter": "crop=iw/2:ih:0:0,v360=input=he:output=fisheye:v_fov=${fov}:h_fov=${fov}:w=${width}:h=${height}",
            "parameter": {
                "width": 1024,
                "height": 1024,
                "fov": 180
            }
        }
        configR = {
            "video_filter": "crop=iw/2:ih:iw/2:0,v360=input=he:output=fisheye:v_fov=${fov}:h_fov=${fov}:w=${width}:h=${height}",
            "parameter": {
                "width": 1024,
                "height": 1024,
                "fov": 180
            }
        }

        frameL = FFmpegStream.get_projection(frame, configL)
        frameR = FFmpegStream.get_projection(frame, configR)
    else:
        # TODO specify size global
        width = 2048
        frame = cv2.resize(frame, (int(width), int(width/2)))
        frameL = frame[:, :int(width/2)]
        frameR = frame[:, int(width/2):]
   
    frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)

    sam2 = GroundingDinoSAM2Segment()
    (_, imgLMask) = sam2.predict([frameL], maskLThreshold, maskLPrompt)
    (_, imgRMask) = sam2.predict([frameR], maskRThreshold, maskRPrompt)

    del sam2
    
    maskL = Image.fromarray((imgLMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
    maskR = Image.fromarray((imgRMask[0].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')

    if False:
        red_transparent = Image.new("RGB", maskL.size, "red")
        red_transparent = red_transparent.convert("RGBA")
        red_transparent.putalpha(128)

        previewL = Image.composite(
            Image.fromarray(frameL).convert("RGBA"),
            red_transparent, 
            maskL
        )
        previewR = Image.composite(
            Image.fromarray(frameR).convert("RGBA"),
            red_transparent,
            maskR
        )

    previewL = Image.composite(
        Image.new("RGB", maskL.size, "blue"),
        Image.fromarray(frameL).convert("RGBA"),
        maskL.point(lambda p: 100 if p > 1 else 0)
    )
    previewR = Image.composite(
        Image.new("RGB", maskR.size, "blue"),
        Image.fromarray(frameR).convert("RGBA"),
        maskR.point(lambda p: 100 if p > 1 else 0)
    )



    return previewL, maskL, previewR, maskR

with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    gr.Markdown('''
        Process:
        1. Upload Your video
        2. Select Video Source Format
        3. Generate Initial Mask with Button
        4. Add Video Mask Job

        Notes: This tool require min 12GB VRAM to run proberly. If you want to generate Inital masks for the next job while a background video mask job is runnung you need 24GB VRAM.
    ''')
    with gr.Column():
        input_video = gr.File(label="Upload Video (MKV or MP4)", file_types=["mkv", "mp4", "video"])
        projection_dropdown = gr.Dropdown(choices=["eq", "fisheye180", "fisheye190", "fisheye200"], label="VR Video Source Format", value="eq")
    

    mask_button = gr.Button("1. Generate Initial Mask")
    with gr.Row():
        with gr.Column():
            maskLPrompt = gr.Textbox(label="Left Eye Mask Prompt", value="top person", lines=1, max_lines=1)
            maskLThreshold = gr.Number(
                label="Threshold",
                minimum=0.01,
                maximum=0.99,
                step=0.01,
                value=0.3
            )
        with gr.Column():
            maskRPrompt = gr.Textbox(label="Right Eye Mask Prompt", value="top person", lines=1, max_lines=1)
            maskRThreshold = gr.Number(
                label="Threshold",
                minimum=0.01,
                maximum=0.99,
                step=0.01,
                value=0.3
            )
    with gr.Row():
        previewL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
        previewR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
    with gr.Row():
        maskL = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')
        maskR = gr.Image(value=None, type='numpy', format='png', image_mode='RGB')

    mask_button.click(
        fn=get_mask,
        inputs=[input_video, projection_dropdown, maskLPrompt, maskRPrompt, maskLThreshold, maskRThreshold],
        outputs=[previewL, maskL, previewR, maskR]
    )

    add_button = gr.Button("2. Add Job")
    status = gr.Textbox(label="Status", lines=2)

    mask_videos = gr.File(value=[], label="Download Mask Videos", visible=True)
    output_videos = gr.File(value=[], label="Download AR Videos", visible=True)

    restart_button = gr.Button("Clear all jobs and data".upper())

    add_button.click(
        fn=add_job,
        inputs=[input_video, projection_dropdown, maskL, maskR],
        outputs=[input_video, previewL, maskL, previewR, maskR]
    )

    restart_button.click(
        # dirty hack, we use k8s restart pod
        fn=lambda: os.system("pkill python"),
        inputs=[],
        outputs=[]
    )

    timer1 = gr.Timer(2, active=True)
    timer5 = gr.Timer(5, active=True)
    timer1.tick(status_text, outputs=status)
    timer5.tick(lambda: result_list, outputs=output_videos)
    timer5.tick(lambda: mask_list, outputs=mask_videos)
    demo.load(fn=status_text, outputs=status)
    demo.load(fn=lambda: mask_list, outputs=mask_videos)
    demo.load(fn=lambda: result_list, outputs=output_videos)


if __name__ == "__main__":
    print("gradio version", gr.__version__)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("exit")
