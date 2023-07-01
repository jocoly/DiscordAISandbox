import {getNumImages} from "../tools/getNumImages.js";
import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";

export async function dreamlikePhotoreal(msg) {
    const enqueueMessage = await msg.reply("Request added to the queue. There are " + queue.length + " requests ahead of you.");
    const numImages = await getNumImages(msg);
    if (numImages > process.env.MAX_NUM_IMAGES) {
        await msg.reply("That's too many images. Try requesting fewer.")
    } else {
        queue.push({msg, pipeline: 'DreamlikePhotoreal', enqueueMessageId: enqueueMessage.id});
    }
    if (queue.length === 1) {
        await processQueue();
    }
}