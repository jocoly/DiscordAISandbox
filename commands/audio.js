import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";

export async function audio(msg) {
    const enqueueMessage = await msg.reply("Request added to the queue. There are " + queue.length + " requests ahead of you.");
    queue.push({msg, pipeline: 'TextToAudio', enqueueMessageId: enqueueMessage.id})
    if (queue.length === 1) {
        await processQueue();
    }
}