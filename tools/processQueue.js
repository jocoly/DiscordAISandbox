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
        let enqueueMessage;
        let confirmationMessage;
        let answer;
        try {
            enqueueMessage = await msg.channel.messages.fetch(queue[0].enqueueMessageId);
        } catch (error) {
            console.log("Error fetching queue message: " + error)
        }
        try {
            await enqueueMessage.delete();
        } catch (error) {
            console.log("Error deleting enqueue message:" + error);
        }
        if (queue[0].pipeline !== "Chat") {
            confirmationMessage = await msg.reply('Processing your request...');
        }
        let isReply;
        try {
            refMsg = await msg.fetchReference();
            isReply = true;
        } catch (error) {
            isReply = false;
        }
        if ((msg.content.includes("!upscale") || msg.content.includes("!img2img") || msg.content.includes("!caption")) && isReply) {
            prompt = getPrompt(refMsg);
            numImages = 1;
            if (refMsg.attachments.size > 0) {
                try {
                    imageUrl = Array.from(refMsg.attachments.values())[0].url;
                } catch (error) {
                    console.log("Error getting attachment url:" + error);
                    await msg.reply("Error retrieving attachment. Try again later.");
                }
            }
        } else {
            if ((msg.content.includes("!upscale") || msg.content.includes("!img2img") || msg.content.includes("!caption")) && !isReply) {
                prompt = getPrompt(msg);
                numImages = 1;
                if (msg.attachments.size > 0) {
                    try {
                        imageUrl = Array.from(msg.attachments.values())[0].url;
                    } catch (error) {
                        console.log("Error getting attachment url:" + error);
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
            if (queue[0].pipeline === "Chat") {
                console.log(results[0])
                answer = results[0].slice(6)
                answer = answer.replace(/....$/,'')
                await msg.reply(answer)
            } else if (queue[0].pipeline === "Caption") {
                await msg.reply(results[0])
            } else {
                await msg.reply({files: results, content: getPrompt(msg)});
            }
        } catch (error) {
            console.log("Error sending reply: " + error)
            await msg.reply("Internal server error. Try again later.")
        }
        try {
            if (queue[0].pipeline !== "Chat"){
                await confirmationMessage.delete();
            }
        } catch (error) {
            console.log("Error deleting confirmation message: " + error)
        }
        try {
            if (process.env.DELETE_AFTER_SENDING === 'true' && queue[0].pipeline !== "Chat"&& queue[0].pipeline !== "Caption") {
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