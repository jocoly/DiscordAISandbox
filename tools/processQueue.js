import {getNumImages} from "./getNumImages.js";
import {callBackendPipeline} from "./backendAPI.js";
import {getPrompt} from "./getPrompt.js";
import fs from "fs";
import {client, queue} from "../bot.js";

export async function processQueue() {
    while (queue.length > 0) {
        const msg = queue[0].msg;
        const enqueueMessage = await msg.channel.messages.fetch(queue[0].enqueueMessageId)
        try {
            await enqueueMessage.delete();
        } catch (error) {
            console.log("Error deleting enqueue message:" + error);
        }
        const confirmationMessage = await msg.reply('Processing your request...');
        if (msg.author.id === client.user.id) {
            let msg = await msg.channel.messages.fetch(msg.reference.messageId);
        }
        let numImages;
        if (msg.content.includes('!img2img')){
            numImages = 1;
        } else {
            numImages = getNumImages(msg);
        }
        let imageUrl = ""
        if (msg.attachments.size > 0) {
            try {
                imageUrl = Array.from(msg.attachments.values())[0].url
            } catch (error) {
                console.log("Error getting attachment url:" + error)
                await msg.reply("Error retrieving attachment. Try again later.")
            }
        }
        try {
            const results = await callBackendPipeline(getPrompt(msg), queue[0].pipeline, numImages, imageUrl);
            await msg.reply({files: results, content: getPrompt(msg)});
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