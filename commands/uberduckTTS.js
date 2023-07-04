import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";
import {getCommand} from "../tools/getCommand.js";

export async function uberduckTTS(msg) {
    const enqueueMessage = await msg.reply("Request added to the queue. There are " + queue.length + " requests ahead of you.");
    queue.push({msg, pipeline: getCommand(msg), enqueueMessageId: enqueueMessage.id})
    if (queue.length === 1) {
        await processQueue();
    }
}