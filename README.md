# Discord AI Sandbox

## A Discord interface for Stable Diffusion Pipelines

Everything is locally hosted via a backend Python server. You'll need 16 GB minimum VRAM to run the video pipelines.

Settings can be tweaked in a dotenv file. Copy the template below and add your own Discord token to get started.

Make sure your bot 

## Supported commands:

    !test
    -Sends a test response to show that the bot is working.

    !drawX <prompt>
    -Submits the prompt for processing using the Stable Diffusion 2 base model pipeline and replies with the result. Replace 'X' with an integer to specify the number of images to generate (default is 1).

    !video <prompt>
    -Submits the prompt for processing using the Modelscope Text-to-Video pipeline and replies with the result.

    !img2img <prompt>
    -Submits the prompt and the first file attachment from:
        -the reference message (the message being replied to) if the command is sent in a reply
            - or -
        -the message the command is sent in if the command is not in a reply
    ...for processing using the Stable Diffusion Image-to-Image model pipeline and replies with the result.

    !upscale
    -Submits the prompt and the first file attachment from:
        -the reference message (the message being replied to) if the command is sent in a reply
            - or -
        -the message the command is sent in if the command is not in a reply
    ...for processing using the Stable Diffusion Latent Upscale model pipeline and replies with the result.

    !xlvid <prompt>
    -Submits the prompt for processing using the Zeroscope_v2_576w Text-to-Video model pipeline and replies with the result.

    And the following work just like !drawX but use different models:
    -!realistic/!rv: Realistic Vision 2.0
    -!openjourney/!oj: Openjourney
    -!dreamshaper/!ds: Dream Shaper
    -!anything: Anything_v3.0
    -!photoreal/!pr: Dreamlike Photoreal

## To run:

-Clone the repo and copy the following into a new file called '.env' saved in the root directory of the project.

    STABLE_DIFFUSION=true
    TEXT_TO_VIDEO=true
    IMAGE_TO_IMAGE=true
    XL_VIDEO=true
    UPSCALE=true

    IMAGE_INFERENCE_STEPS=50
    IMAGE_GUIDANCE_SCALE=7.5

    IMAGE_WIDTH=512
    IMAGE_HEIGHT=512

    VIDEO_INFERENCE_STEPS=50
    VIDEO_GUIDANCE_SCALE=7.5
    VIDEO_NUM_FRAMES=24

    VIDEO_WIDTH=256
    VIDEO_HEIGHT=256

    IMG2IMG_STRENGTH=0.75
    IMG2IMG_INFERENCE_STEPS=50
    IMG2IMG_GUIDANCE_SCALE=7.5

    UPSCALE_INFERENCE_STEPS=50
    UPSCALE_GUIDANCE_SCALE=7.5

    NEGATIVE_PROMPT=blurry, watermark, gross, disgusting

    OUTPUT_DIR=./output/images/
    BACKEND_ADDRESS=127.0.0.1
    PORT=8001

    CONTAIN_BOT=true

    DISCORD_TOKEN=<TOKEN GOES HERE>
    DISCORD_CHANNEL_ID=1077752711145078854

    DELETE_AFTER_SENDING=true
    MAX_NUM_IMAGES=6

-Install Python requirements
    
    pip install -r "requirements.txt"

-Install Node.js requirements

    npm install

-Start the backend server (will take a while to download the models the first time)
    
    python main.py

-In a new terminal window, start the bot

    node bot.js