import random
import subprocess
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from diffusers.utils import export_to_video
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionLatentUpscalePipeline, \
    DPMSolverMultistepScheduler

load_dotenv('./.env')
app = Flask(__name__)
CORS(app)
print("--> Starting the backend server. This may take some time.")

print("CUDA-enabled gpu detected: " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if (os.getenv("STABLE_DIFFUSION")) == 'true':
    print("Loading Stable Diffusion 2 base model")
    stable_diffusion_pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base',
                                                              torch_dtype=torch.float16,
                                                              variant='fp16')
    stable_diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(stable_diffusion_pipe.scheduler.config)
    stable_diffusion_pipe = stable_diffusion_pipe.to(device)
    stable_diffusion_pipe.enable_model_cpu_offload()
    stable_diffusion_pipe.enable_vae_slicing()

if (os.getenv("TEXT_TO_VIDEO")) == 'true':
    print("Loading Modelscope Text-to-Video model")
    text_to_video_pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b',
                                                           torch_dtype=torch.float16,
                                                           variant='fp16')
    text_to_video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(text_to_video_pipe.scheduler.config)
    text_to_video_pipe = text_to_video_pipe.to(device)
    text_to_video_pipe.enable_model_cpu_offload()
    text_to_video_pipe.enable_vae_slicing()

if (os.getenv("IMAGE_TO_IMAGE")) == 'true':
    print("Loading Image-to-Image model")
    image_to_image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                                         torch_dtype=torch.float16,
                                                                         variant='fp16')
    image_to_image_pipe.scheduler = DPMSolverMultistepScheduler.from_config(image_to_image_pipe.scheduler.config)
    image_to_image_pipe = image_to_image_pipe.to(device)
    image_to_image_pipe.enable_model_cpu_offload()

if (os.getenv("XL_VIDEO")) == 'true':
    print("Loading Modelscope Text-to-Video XL model")
    t2v_xl_pipe = DiffusionPipeline.from_pretrained('cerspense/zeroscope_v2_576w',
                                                    torch_dtype=torch.float16,
                                                    variant='fp16')
    t2v_xl_pipe.scheduler = DPMSolverMultistepScheduler.from_config(t2v_xl_pipe.scheduler.config)
    t2v_xl_pipe = t2v_xl_pipe.to(device)
    t2v_xl_pipe.enable_model_cpu_offload()
    t2v_xl_pipe.enable_vae_slicing()

if (os.getenv("UPSCALE")) == 'true':
    print("Loading Stable Diffusion Latent Upscale model")
    upscale_pipe = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler',
                                                                        torch_dtype=torch.float16,
                                                                        variant='fp16')
    upscale_pipe = upscale_pipe.to(device)

if (os.getenv("REALISTIC_VISION")) == 'true':
    print("Loading Realistic Vision model")
    realistic_vision_pipe = DiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V2.0',
                                                              torch_dtype=torch.float16,
                                                              variant='fp16')
    realistic_vision_pipe.scheduler = DPMSolverMultistepScheduler.from_config(realistic_vision_pipe.scheduler.config)
    realistic_vision_pipe = realistic_vision_pipe.to(device)
    realistic_vision_pipe.enable_model_cpu_offload()
    realistic_vision_pipe.enable_vae_slicing()

if (os.getenv("OPENJOURNEY")) == 'true':
    print("Loading openjourney model")
    openjourney_pipe = DiffusionPipeline.from_pretrained('prompthero/openjourney',
                                                         torch_dtype=torch.float16,
                                                         variant='fp16')
    openjourney_pipe.scheduler = DPMSolverMultistepScheduler.from_config(openjourney_pipe.scheduler.config)
    openjourney_pipe = openjourney_pipe.to(device)
    openjourney_pipe.enable_model_cpu_offload()
    openjourney_pipe.enable_vae_slicing()

if (os.getenv("DREAM_SHAPER")) == 'true':
    print("Loading Dream Shaper model")
    dream_shaper_pipe = DiffusionPipeline.from_pretrained('Lykon/DreamShaper',
                                                          torch_dtype=torch.float16,
                                                          variant='fp16')
    dream_shaper_pipe.scheduler = DPMSolverMultistepScheduler.from_config(dream_shaper_pipe.scheduler.config)
    dream_shaper_pipe = dream_shaper_pipe.to(device)
    dream_shaper_pipe.enable_model_cpu_offload()
    dream_shaper_pipe.enable_vae_slicing()

if (os.getenv("ANYTHING_V3")) == 'true':
    print("Loading Anything-v3.0 model")
    anything_pipe = DiffusionPipeline.from_pretrained('Linaqruf/anything-v3.0',
                                                      torch_dtype=torch.float16,
                                                      variant='fp16')
    anything_pipe.scheduler = DPMSolverMultistepScheduler.from_config(anything_pipe.scheduler.config)
    anything_pipe = anything_pipe.to(device)
    anything_pipe.enable_model_cpu_offload()
    anything_pipe.enable_vae_slicing()

if (os.getenv("DREAMLIKE_PHOTOREAL")) == 'true':
    print("Loading Dreamlike Photoreal model")
    dreamlike_photoreal_pipe = DiffusionPipeline.from_pretrained('dreamlike-art/dreamlike-photoreal-2.0',
                                                                 torch_dtype=torch.float16,
                                                                 variant='fp16')
    dreamlike_photoreal_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        dreamlike_photoreal_pipe.scheduler.config)
    dreamlike_photoreal_pipe = dreamlike_photoreal_pipe.to(device)
    dreamlike_photoreal_pipe.enable_model_cpu_offload()
    dreamlike_photoreal_pipe.enable_vae_slicing()

processing_lock = threading.Lock()


def process(prompt: str, pipeline: str, num: int, img_url: str):
    start_time = time.time()
    print("Processing query...")
    seed = random.randint(0, 100000)
    if torch.cuda.is_available():
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    output_dir = os.getenv("OUTPUT_DIR")
    process_output = []
    match pipeline:
        case "StableDiffusion":
            images_array = stable_diffusion_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case "TextToVideo":
            video_frames = text_to_video_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_frames=int(os.getenv("VIDEO_NUM_FRAMES")),
                num_inference_steps=int(os.getenv("VIDEO_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("VIDEO_GUIDANCE_SCALE")),
                width=int(os.getenv("VIDEO_WIDTH")),
                height=int(os.getenv("VIDEO_HEIGHT")),
                generator=generator,
            ).frames
            gif_file_path = save_frames(video_frames, output_dir)
            process_output.append(gif_file_path)
        case "ImageToImage":
            response = requests.get(img_url)
            input_image = Image.open(BytesIO(response.content)).convert("RGB")
            input_image = input_image.resize((768, 512))
            image_array = image_to_image_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                image=input_image,
                num_inference_steps=int(os.getenv("IMG2IMG_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMG2IMG_GUIDANCE_SCALE")),
                strength=float(os.getenv("IMG2IMG_STRENGTH")),
            ).images
            image_path = save_image(image_array[0], output_dir)
            process_output.append(image_path)
        case "t2vxl":
            video_frames = t2v_xl_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_frames=int(os.getenv("VIDEO_NUM_FRAMES")),
                num_inference_steps=int(os.getenv("VIDEO_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("VIDEO_GUIDANCE_SCALE")),
                width=576,
                height=320,
                generator=generator,
            ).frames
            gif_file_path = save_frames(video_frames, output_dir)
            process_output.append(gif_file_path)
        case "Upscale":
            response = requests.get(img_url)
            input_image = Image.open(BytesIO(response.content)).convert("RGB")
            image_array = upscale_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                image=input_image,
                num_inference_steps=int(os.getenv("UPSCALE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("UPSCALE_GUIDANCE_SCALE")),
                generator=generator,
            ).images
            image_path = save_image(image_array[0], output_dir)
            process_output.append(image_path)
        case "RealisticVision":
            images_array = realistic_vision_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case "Openjourney":
            images_array = openjourney_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case "DreamShaper":
            images_array = dream_shaper_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case "Anything":
            images_array = anything_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image_spoiler(images_array[index], output_dir)
                process_output.append(image_path)
        case "DreamlikePhotoreal":
            images_array = dreamlike_photoreal_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("IMAGE_WIDTH")),
                height=int(os.getenv("IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image_spoiler(images_array[index], output_dir)
                process_output.append(image_path)

    gen_time = time.time() - start_time
    print(f"Created generation in {gen_time} seconds")
    return process_output


@app.route("/process", methods=["POST"])
def process_api():
    json_data = request.get_json(force=True)
    text_prompt = json_data["text_prompt"]
    pipeline = json_data["pipeline"]
    num = int(json_data["num"])
    image_url = json_data["image_url"]
    with processing_lock:
        generation = process(text_prompt, pipeline, num, image_url)
    response = {'generation': generation}
    return jsonify(response)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify(success=True)


def save_frames(video_frames, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = str(uuid.uuid4()) + '.mp4'
    mp4_file_path = os.path.join(output_dir, file_name)
    export_to_video(video_frames, mp4_file_path)
    gif_file_path = convert_to_gif(mp4_file_path)
    os.remove(mp4_file_path)
    return gif_file_path


def save_image(image, output_dir):
    file_name = str(uuid.uuid4()) + '.png'
    image_path = os.path.join(output_dir, file_name)
    image.save(image_path, format='png')
    return image_path


def save_image_spoiler(image, output_dir):
    filename = "SPOILER_" + str(random.randint(1000, 9999)) + ".png"
    image_path = os.path.join(output_dir, filename)
    image.save(image_path, format='png')
    return image_path


def convert_to_gif(mp4_file_path):
    gif_file_path = mp4_file_path[:-4] + ".gif"
    subprocess.run(['ffmpeg', '-i', mp4_file_path, '-vf', 'fps=10,scale=320:-1:flags=lanczos', gif_file_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return gif_file_path


if __name__ == "__main__":
    app.run(host=os.getenv("BACKEND_ADDRESS"), port=os.getenv("PORT"), debug=False)
