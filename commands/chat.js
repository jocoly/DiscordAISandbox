import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";
export async function chat(msg) {
    queue.push({msg, pipeline: 'Chat'});
    if (queue.length === 1) {
        await processQueue();
    }
}