import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";

export async function caption(msg) {
    const enqueueMessage = await msg.reply("Request added to the queue. There are " + queue.length + " requests ahead of you.");
    queue.push({msg, pipeline: "Caption", enqueueMessageId: enqueueMessage.id})
    if (queue.length === 1) {
        await processQueue();
    }
}