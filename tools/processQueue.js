import {getNumImages} from "./getNumImages.js";
import {callBackendPipeline} from "./backendAPI.js";
import {getPrompt} from "./getPrompt.js";
import fs from "fs";
import {client, queue} from "../bot.js";

export async function processQueue() {
    while (queue.length > 0) {
        let init_msg = queue[0].msg;
        let msg;
        let prompt;
        let promptMsg;
        const enqueueMessage = await init_msg.channel.messages.fetch(queue[0].enqueueMessageId)
        try {
            await enqueueMessage.delete();
        } catch (error) {
            console.log("Error deleting enqueue message:" + error);
        }
        const confirmationMessage = await init_msg.reply('Processing your request...');
        if (init_msg.content.includes("!upscalevid")) {
            msg = await init_msg.channel.messages.fetch(init_msg.reference.messageId);
            promptMsg = await msg.channel.messages.fetch(msg.reference.messageId);
            prompt = getPrompt(promptMsg);
        } else {
            msg = init_msg;
            prompt = getPrompt(msg);
        }
        let numImages;
        let imageUrl;
        let results;
        if (msg.content.includes('!img2img')){
            numImages = 1;
            if (msg.attachments.size > 0) {
            try {
                imageUrl = Array.from(msg.attachments.values())[0].url
            } catch (error) {
                console.log("Error getting attachment url:" + error)
                await msg.reply("Error retrieving attachment. Try again later.")
            }
        }
        } else {
            numImages = getNumImages(msg);
            imageUrl = ""
        }
        try {
            console.log(getPrompt(msg))
            results = await callBackendPipeline(prompt, queue[0].pipeline, numImages, imageUrl);
        } catch (error) {
            console.log("Error getting results from backend: " + error)
        }
        try {
            await msg.reply({files: results, content: getPrompt(msg)});
        } catch (error) {
            console.log("Error sending reply: " + error)
        }
        try {
            await confirmationMessage.delete();
        } catch (error) {
            console.log("Error deleting confirmation message: " + error)
        }
        try {
            if (process.env.DELETE_AFTER_SENDING == 'true') {
                for (const result of results) {
                    await fs.unlinkSync(result)
                }
            }
        } catch (error) {
            console.log("Error deleting file(s): " + error)
        }
            queue.shift();
    }
}