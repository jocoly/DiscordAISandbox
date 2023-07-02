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