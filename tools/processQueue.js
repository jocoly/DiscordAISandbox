import {getNumImages} from "./getNumImages.js";
import {callBackendPipeline} from "./backendAPI.js";
import {getPrompt} from "./getPrompt.js";
import fs from "fs";
import {queue} from "../bot.js";

export async function processQueue() {
    while (queue.length > 0) {
        const msg = queue[0].msg;
        const enqueueMessage = await msg.channel.messages.fetch(queue[0].enqueueMessageId)
        if (enqueueMessage) {
            try {
                await enqueueMessage.delete();
            } catch (error) {
                console.log("Error deleting enqueueMessage:" + error);
            }
        }
        const confirmationMessage = await msg.reply('Processing your request...');
        const numImages = getNumImages(msg)
        try {
            const results = await callBackendPipeline(getPrompt(msg), queue[0].pipeline, numImages);
            await msg.reply({files: results});
            await confirmationMessage.delete();
            if (process.env.DELETE_AFTER_SENDING == 'true') {
                for (const result of results) {
                    await fs.unlinkSync(result)
                }
            }
            queue.shift();
        } catch (error) {
            console.log("Error connecting to backend server:" + error)
        }
    }
}