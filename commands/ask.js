import {queue} from "../bot.js";
import {processQueue} from "../tools/processQueue.js";
export async function ask(msg) {
    queue.push({msg, pipeline: 'Ask'});
    if (queue.length === 1) {
        await processQueue();
    }
}