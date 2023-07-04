import {openai} from '../bot.js';
import {getPrompt} from "../tools/getPrompt.js";


export async function chat(msg) {
    let prompt = getPrompt(msg);
    let completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: prompt,
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: Number(process.env.CHAT_TEMPERATURE)
    });
    try {
        await msg.reply(prompt + "\n" + completion.data.choices[0].text);
    } catch (error) {
        console.log("Error getting completion: " + error);
        await msg.reply("Error getting completion. Try again later.");
    }
}

export async function prompt(msg) {
    let userPrompt = getPrompt(msg);
    let completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "You will now act as a prompt generator. I will describe an image or a topic to you, and you will create a prompt that could be used for image-generation. The image I want to generate is: " + userPrompt,
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: Number(process.env.CHAT_TEMPERATURE)
    })
    try {
        await msg.reply(completion.data.choices[0].text);
    } catch (error) {
        console.log("Error getting completion: " + error);
        await msg.reply("Error getting completion. Try again later.");
    }
}

