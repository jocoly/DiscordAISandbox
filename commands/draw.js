import {getNumImages} from "../tools/getNumImages.js";

export async function draw(msg) {
    const numImages = await getNumImages(msg);
    if (numImages > process.env.MAX_NUM_IMAGES) {
        await msg.reply("That's too many images. Try requesting fewer.")
    } else {
        queue.push({msg, pipeline: 'StableDiffusion'});
    }
    if (queue.length === 1) {
        await processQueue();
    }
}