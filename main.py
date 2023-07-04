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
import soundfile as sf
from datasets import load_dataset
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionLatentUpscalePipeline, \
    DPMSolverMultistepScheduler, AudioLDMPipeline
import scipy
from transformers import AutoProcessor, SpeechT5HifiGan, SpeechT5ForTextToSpeech, T5Tokenizer, \
    T5ForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

load_dotenv('./.env')
app = Flask(__name__)
CORS(app)
print("--> Starting the backend server. This may take some time.")

print("CUDA-enabled gpu detected: " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if (os.getenv("ASK")) == 'true':
    print("Loading Google FLAN-T5 language model")
    ask_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    ask_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",
                                                           device_map="auto",
                                                           torch_dtype=torch.float16)

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

if (os.getenv("TEXT_TO_AUDIO")) == 'true':
    print("Loading Audio Latent Diffusion model")
    text_to_audio_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2",
                                                          torch_dtype=torch.float16,
                                                          variant='fp16')
    text_to_audio_pipe = text_to_audio_pipe.to(device)
    text_to_audio_pipe.enable_sequential_cpu_offload()
    text_to_audio_pipe.enable_vae_slicing()

if (os.getenv("TEXT_TO_SPEECH")) == 'true':
    print("Loading Text-to-Speech model")
    processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts",
                                              torch_dtype=torch.float16,
                                              variant='fp16')
    text_to_speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

if (os.getenv("UBERDUCK_TTS")) == 'true':
    uberduck_auth = (os.getenv("UBERDUCK_API_KEY"), os.getenv("UBERDUCK_API_SECRET"))

if (os.getenv("CAPTION")) == 'true':
    print("Loading image-captioning model")
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_model.to(device)

if (os.getenv("IMAGE_TO_IMAGE")) == 'true':
    print("Loading Image-to-Image model")
    image_to_image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                                         torch_dtype=torch.float16,
                                                                         variant='fp16')
    image_to_image_pipe.scheduler = DPMSolverMultistepScheduler.from_config(image_to_image_pipe.scheduler.config)
    image_to_image_pipe = image_to_image_pipe.to(device)
    image_to_image_pipe.enable_model_cpu_offload()

if (os.getenv("ANIMOV_512X")) == 'true':
    print("Loading animov-512x model")
    animov_pipe = DiffusionPipeline.from_pretrained('strangeman3107/animov-512x',
                                                           torch_dtype=torch.float16,
                                                           variant='fp16')
    animov_pipe.scheduler = DPMSolverMultistepScheduler.from_config(animov_pipe.scheduler.config)
    animov_pipe = animov_pipe.to(device)
    animov_pipe.enable_model_cpu_offload()
    animov_pipe.enable_vae_slicing()

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

if (os.getenv("WAIFU_DIFFUSION")) == 'true':
    print("Loading Waifu Diffusion model")
    waifu_diffusion_pipe = DiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',
                                                             torch_dtype=torch.float32,
                                                             variant='fp32')
    waifu_diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(waifu_diffusion_pipe.scheduler.config)
    waifu_diffusion_pipe = waifu_diffusion_pipe.to(device)
    waifu_diffusion_pipe.enable_model_cpu_offload()
    waifu_diffusion_pipe.enable_vae_slicing()

if (os.getenv("VOX2")) == 'true':
    print("Loading vox2 model")
    vox2_pipe = DiffusionPipeline.from_pretrained('plasmo/vox2',
                                                  torch_dtype=torch.float16,
                                                  variant='fp16')
    vox2_pipe.scheduler = DPMSolverMultistepScheduler.from_config(vox2_pipe.scheduler.config)
    vox2_pipe = vox2_pipe.to(device)
    vox2_pipe.enable_model_cpu_offload()
    vox2_pipe.enable_vae_slicing()

processing_lock = threading.Lock()


def process(prompt: str, pipeline: str, num: int, img_url: str):
    global output_ready
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
        case "Ask":
            input_ids = ask_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            outputs = ask_model.generate(input_ids)
            output_string = ask_tokenizer.decode(outputs[0])
            process_output.append(output_string)
        case "StableDiffusion":
            images_array = stable_diffusion_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("SD_IMAGE_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("SD_IMAGE_GUIDANCE_SCALE")),
                width=int(os.getenv("SD_IMAGE_WIDTH")),
                height=int(os.getenv("SD_IMAGE_HEIGHT")),
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
                width=256,
                height=256,
                generator=generator,
            ).frames
            gif_file_path = save_frames(video_frames, output_dir)
            process_output.append(gif_file_path)
        case "TextToAudio":
            audio = text_to_audio_pipe(
                prompt=prompt,
                num_inference_steps=int(os.getenv("AUDIO_INFERENCE_STEPS")),
                audio_length_in_s=float(os.getenv("AUDIO_LENGTH_IN_SECONDS")),
            ).audios[0]
            wav_file_path = save_audio(audio, output_dir)
            process_output.append(wav_file_path)
        case "TextToSpeech":
            inputs = processor(
                text=prompt,
                return_tensors="pt"
            )
            speech = text_to_speech_model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            file_name = os.path.join(output_dir, str(random.randint(1111, 9999)) + ".wav")
            sf.write(file_name, speech.numpy(), samplerate=16000)
            process_output.append(file_name)
        case "Caption":
            max_length = 16
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
            response = requests.get(img_url)
            input_image = Image.open(BytesIO(response.content)).convert("RGB")
            images_array = [input_image]
            pixel_values = feature_extractor(images=images_array, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(device)
            output_ids = caption_model.generate(pixel_values, **gen_kwargs)
            preds = caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            process_output.append(preds[0])
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
        case "animov":
            video_frames = animov_pipe(
                prompt=prompt + " - anime",
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_frames=int(os.getenv("VIDEO_NUM_FRAMES")),
                num_inference_steps=int(os.getenv("VIDEO_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("VIDEO_GUIDANCE_SCALE")),
                width=256,
                height=256,
                generator=generator,
            ).frames
            gif_file_path = save_frames(video_frames, output_dir)
            process_output.append(gif_file_path)
        case "Upscale":
            response = requests.get(img_url)
            input_image = Image.open(BytesIO(response.content)).convert("RGB")
            image_array = upscale_pipe(
                image=input_image,
                generator=generator,
            ).images
            image_path = save_image(image_array[0], output_dir)
            process_output.append(image_path)
        case "RealisticVision":
            images_array = realistic_vision_pipe(
                prompt=prompt + "(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, "
                                "Fujifilm XT3",
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("RV_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("RV_GUIDANCE_SCALE")),
                width=int(os.getenv("RV_IMAGE_WIDTH")),
                height=int(os.getenv("RV_IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case "Openjourney":
            images_array = openjourney_pipe(
                prompt="mdjrny-v4 style " + prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("OJ_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("OJ_GUIDANCE_SCALE")),
                width=int(os.getenv("OJ_IMAGE_WIDTH")),
                height=int(os.getenv("OJ_IMAGE_HEIGHT")),
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
                num_inference_steps=int(os.getenv("DS_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("DS_GUIDANCE_SCALE")),
                width=int(os.getenv("DS_IMAGE_WIDTH")),
                height=int(os.getenv("DS_IMAGE_HEIGHT")),
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
                num_inference_steps=int(os.getenv("ANYTHING_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("ANYTHING_GUIDANCE_SCALE")),
                width=int(os.getenv("ANYTHING_IMAGE_WIDTH")),
                height=int(os.getenv("ANYTHING_IMAGE_HEIGHT")),
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
                num_inference_steps=int(os.getenv("PR_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("PR_GUIDANCE_SCALE")),
                width=int(os.getenv("PR_IMAGE_WIDTH")),
                height=int(os.getenv("PR_IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image_spoiler(images_array[index], output_dir)
                process_output.append(image_path)
        case "WaifuDiffusion":
            images_array = waifu_diffusion_pipe(
                prompt=prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("WD_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("WD_GUIDANCE_SCALE")),
                width=int(os.getenv("WD_IMAGE_WIDTH")),
                height=int(os.getenv("WD_IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image_spoiler(images_array[index], output_dir)
                process_output.append(image_path)
        case "vox2":
            images_array = vox2_pipe(
                prompt="voxel-ish, intricate detail: " + prompt,
                negative_prompt=os.getenv("NEGATIVE_PROMPT"),
                num_images_per_prompt=num,
                num_inference_steps=int(os.getenv("VOX2_INFERENCE_STEPS")),
                guidance_scale=float(os.getenv("VOX2_GUIDANCE_SCALE")),
                width=int(os.getenv("VOX2_IMAGE_WIDTH")),
                height=int(os.getenv("VOX2_IMAGE_HEIGHT")),
                generator=generator,
            ).images
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for index in range(num):
                image_path = save_image(images_array[index], output_dir)
                process_output.append(image_path)
        case _:
            voice_model_uuid = pipeline
            audio_uuid = requests.post(
                "https://api.uberduck.ai/speak",
                json=dict(speech=prompt, voicemodel_uuid=voice_model_uuid),
                auth=uberduck_auth,
            ).json()["uuid"]
            output = requests.get(
                "https://api.uberduck.ai/speak-status",
                params=dict(uuid=audio_uuid),
                auth=uberduck_auth,
            ).json()
            for t in range(10):
                time.sleep(1)  # check status every second for 10 seconds.
                output = requests.get(
                    "https://api.uberduck.ai/speak-status",
                    params=dict(uuid=audio_uuid),
                    auth=uberduck_auth,
                ).json()
                if "path" in output:
                    time.sleep(3)
                    output_ready = True
                    break

            if output_ready:
                print(output["path"])
                r = requests.get(output["path"], allow_redirects=True)
                file_name = os.path.join(output_dir, str(random.randint(1111, 9999)) + ".wav")
                with open(file_name, 'wb') as file:
                    file.write(r.content)
                process_output.append(file_name)
            else:
                print("Audio file generation failed")

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
    filename = "SPOILER_" + str(random.randint(10000, 99999)) + ".png"
    image_path = os.path.join(output_dir, filename)
    image.save(image_path, format='png')
    return image_path


def save_audio(audio, output_dir):
    file_name = str(uuid.uuid4()) + '.wav'
    wav_file_path = os.path.join(output_dir, file_name)
    scipy.io.wavfile.write(wav_file_path, rate=16000, data=audio)
    return wav_file_path


def convert_to_gif(mp4_file_path):
    gif_file_path = mp4_file_path[:-4] + ".gif"
    subprocess.run(['ffmpeg', '-i', mp4_file_path, '-vf', 'fps=10,scale=320:-1:flags=lanczos', gif_file_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return gif_file_path


if __name__ == "__main__":
    app.run(host=os.getenv("BACKEND_ADDRESS"), port=os.getenv("PORT"), debug=False)
