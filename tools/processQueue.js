import {getNumImages} from "./getNumImages.js";
import {callBackendPipeline} from "./backendAPI.js";
import {getPrompt} from "./getPrompt.js";
import fs from "fs";
import {queue} from "../bot.js";

export async function processQueue() {
    while (queue.length > 0) {
        let msg = queue[0].msg;
        let prompt;
        let refMsg;
        let numImages;
        let imageUrl;
        let results;
        const enqueueMessage = await msg.channel.messages.fetch(queue[0].enqueueMessageId)
        try {
            await enqueueMessage.delete();
        } catch (error) {
            console.log("Error deleting enqueue message:" + error);
        }
        const confirmationMessage = await msg.reply('Processing your request...');
        let isReply;
        try {
            refMsg = await msg.fetchReference()
            isReply = true;
        } catch (error) {
            isReply = false;
        }
        if ((msg.content.includes("!upscale") || msg.content.includes("!img2img")) && isReply) {
            prompt = getPrompt(refMsg)
            numImages = 1;
            if (refMsg.attachments.size > 0) {
                try {
                    imageUrl = Array.from(refMsg.attachments.values())[0].url
                } catch (error) {
                    console.log("Error getting attachment url:" + error)
                    await msg.reply("Error retrieving attachment. Try again later.")
                }
            }
        } else {
            if (msg.content.includes("!upscale") || msg.content.includes("!img2img") && !isReply) {
                prompt = getPrompt(msg);
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
                prompt = getPrompt(msg);
                numImages = getNumImages(msg);
                imageUrl = "";
            }
        }
        try {
            console.log(prompt)
            results = await callBackendPipeline(prompt, queue[0].pipeline, numImages, imageUrl);
        } catch (error) {
            console.log("Error getting results from backend: " + error)
        }
        try {
            await msg.reply({files: results, content: getPrompt(msg)});
        } catch (error) {
            console.log("Error sending reply: " + error)
            await msg.reply("Internal server error. Try again later.")
        }
        try {
            await confirmationMessage.delete();
        } catch (error) {
            console.log("Error deleting confirmation message: " + error)
        }
        try {
            if (process.env.DELETE_AFTER_SENDING === 'true') {
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