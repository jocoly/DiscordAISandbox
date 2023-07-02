# Discord AI Sandbox

## A Discord interface for some of Huggingface's most-downloaded AI image and video generation models

Everything is locally hosted via a backend Python server. You'll need 16 GB minimum VRAM to run the video pipelines. Everything else can run on CPU, but not very quickly.

Settings can be tweaked in a dotenv file. Copy the template below and add your own Discord token to get started.

Make sure your bot has Guilds and GuildMessages intents enabled in the server you want to use it in.

### Demos use the following prompt:
`A portrait of a wise wizard, adorned in ornate robes and holding a crackling staff, emanating an aura of ancient magic.`

## Commands:

`!test`
-Sends a test response to show that the bot is working.

`!ask`
-Sends a Google T5 text prompt and replies with the result.

`!chat`
-Sends a Chat GPT completion prompt and replies with the result.

`!drawX <prompt>`
-Submits the prompt for processing using the Stable Diffusion 2 base model pipeline and replies with the result. Replace 'X' with an integer to specify the number of images to generate (default is 1).

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/37319429-97ee-4cc4-aa6c-df4e826972f5)

`!video <prompt>`
-Submits the prompt for processing using the Modelscope Text-to-Video pipeline and replies with the result.

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/2b909e1c-f68b-4041-9e6a-5d33420d9fdf)

![5d1132d6-587c-4175-a1a7-f1a6f37baf49](https://github.com/jocoly/DiscordAISandbox/assets/62028785/c51bf22f-0709-41fb-960a-a4f61807189f)

`!audio <prompt>`
-Submits the prompt for processing using the Audio Latent Diffusion model and replies with the result.

`!speech <prompt>`
-Submits the prompt for processing using the Microsoft text-to-speech model and replies with the result.

`!img2img <prompt>`

-Submits the prompt and the first attachment for processing using the Stable Diffusion Image-to-Image model pipeline if the command is sent as a standalone message.

-Submits the prompt with the first attachment from the reference message if the command is sent as a reply.

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/fad818fb-2fd7-49b5-943b-66400df780d9)

`!upscale`

-Submits the first attachment for processing using the Stable Diffusion 2x upscale model pipeline if the command is sent as a standalone message.

-Submits the first attachment from the reference message if the command is sent as a reply.

Input (512x512):

![frog1](https://github.com/jocoly/DiscordAISandbox/assets/62028785/8aad69bb-73d5-4ff6-a45c-8d9b6ffa1918)

Result (1024x1024):

![8b6fa288-9609-418a-bbf8-57921ed1b527](https://github.com/jocoly/DiscordAISandbox/assets/62028785/0d3d61b4-b84c-45b0-8cde-17980be11b3a)


`!xlvid <prompt>`
-Submits the prompt for processing using the Zeroscope_v2_576w Text-to-Video model pipeline and replies with the result.

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/ca71f9ef-6b94-49bc-a5d5-e787bb49d06a)

![d51f95e9-3a54-4714-98a2-7df5b20c9810](https://github.com/jocoly/DiscordAISandbox/assets/62028785/764ed152-c85d-4014-be21-7b4baa9cbcc0)

`!caption`
-Submits the first attachment for processing using the GPT-2 image caption pipeline if the command is sent as a standalone message.

Submits the first attachment from the reference message if the command is sent as a reply.

And the following work just like !drawX but use different models:

`!realistic/!rv`
-Realistic Vision 2.0

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/c8df23cc-8c39-415a-b92a-859a74ebefcf)

    
`!openjourney/!oj`
-Openjourney

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/dd8660bb-f699-40eb-9a3c-e2c377098908)

    
`!dreamshaper/!ds`
-Dream Shaper

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/cec6262a-1945-4b4e-b272-b5e2f67a6061)

    
`!anything`
-Anything_v3.0

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/c1bc414a-b555-47a2-a7a5-c2b152734a56)

    
`!photoreal/!pr`
-Dreamlike Photoreal

![image](https://github.com/jocoly/DiscordAISandbox/assets/62028785/02142ff9-d150-4999-bf64-ebe38114de10)


## To run:

-Clone the repo and copy the following into a new file called '.env' saved in the root directory of the project.

    DISCORD_TOKEN=<Your token here>
    DISCORD_CHANNEL_ID=<Channel ID if CONTAIN_BOT==true>

    OUTPUT_DIR=./output/images/
    BACKEND_ADDRESS=127.0.0.1
    PORT=8001
    
    CONTAIN_BOT=true
    DELETE_AFTER_SENDING=true
    MAX_NUM_IMAGES=6
    
    CHAT=true
    STABLE_DIFFUSION=true
    TEXT_TO_VIDEO=true
    TEXT_TO_AUDIO=true
    TEXT_TO_SPEECH=true
    IMAGE_TO_IMAGE=true
    CAPTION=true
    XL_VIDEO=true
    UPSCALE=true
    REALISTIC_VISION=true
    OPENJOURNEY=true
    DREAM_SHAPER=true
    ANYTHING_V3=true
    DREAMLIKE_PHOTOREAL=true
    
    SD_IMAGE_INFERENCE_STEPS=50
    SD_IMAGE_GUIDANCE_SCALE=7.5
    SD_IMAGE_WIDTH=512
    SD_IMAGE_HEIGHT=512
    
    VIDEO_INFERENCE_STEPS=50
    VIDEO_GUIDANCE_SCALE=7.5
    VIDEO_NUM_FRAMES=24
    VIDEO_WIDTH=256
    VIDEO_HEIGHT=256
    
    AUDIO_INFERENCE_STEPS=10
    AUDIO_LENGTH_IN_SECONDS=5.0
    
    IMG2IMG_INFERENCE_STEPS=50
    IMG2IMG_GUIDANCE_SCALE=7.5
    IMG2IMG_STRENGTH=0.75
    
    UPSCALE_INFERENCE_STEPS=50
    UPSCALE_GUIDANCE_SCALE=7.5
    
    RV_INFERENCE_STEPS=50
    RV_GUIDANCE_SCALE=7.5
    RV_IMAGE_WIDTH=512
    RV_IMAGE_HEIGHT=512
    
    OJ_INFERENCE_STEPS=50
    OJ_GUIDANCE_SCALE=7.5
    OJ_IMAGE_WIDTH=512
    OJ_IMAGE_HEIGHT=512
    
    DS_INFERENCE_STEPS=50
    DS_GUIDANCE_SCALE=7.5
    DS_IMAGE_WIDTH=512
    DS_IMAGE_HEIGHT=512
    
    ANYTHING_INFERENCE_STEPS=50
    ANYTHING_GUIDANCE_SCALE=7.5
    ANYTHING_IMAGE_WIDTH=512
    ANYTHING_IMAGE_HEIGHT=512
    
    PR_INFERENCE_STEPS=50
    PR_GUIDANCE_SCALE=7.5
    PR_IMAGE_WIDTH=768
    PR_IMAGE_HEIGHT=768
    
    NEGATIVE_PROMPT=blurry, watermark, gross, disgusting, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck


-Install Python requirements
    
    pip install -r "requirements.txt"

-Install Node.js requirements

    npm install

-Start the backend server (will take a while to download the models the first time)
    
    python main.py

-In a new terminal window, start the bot

    node bot.js
