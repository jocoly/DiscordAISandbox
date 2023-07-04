import {openai} from "../bot.js";
import {getPrompt} from "../tools/getPrompt.js";
import {getNumImages} from "../tools/getNumImages.js";

export async function getTopic(msg) {
    let userTopic = getPrompt(msg).replace(/\s+/g, '');
    if (userTopic === "") {
        let completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "Name a subject or scene that would make for an interesting image: ",
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: 1.5
    })
        userTopic = completion.data.choices[0].text
    }
    return userTopic
}

export async function random(msg) {
    let topic = await getTopic(msg) + ".";
    let numImages = await getNumImages(msg);
    console.log(topic);
    const imageModelCommands = [
        "!animov",
        "!anything",
        "!draw",
        "!pr",
        "!ds",
        "!oj",
        "!rv",
        "!video",
        "!vox",
        "!wd",
        "!xlvid"
    ]
    const command = imageModelCommands[Math.floor(Math.random()*imageModelCommands.length)]
    const completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "You will now act as an image prompt generator. I will describe an image to you, and you will create a prompt that could be used for image-generation. The image I want to generate is: " + topic,
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: Number(process.env.CHAT_TEMPERATURE)
    })
    let prompt = completion.data.choices[0].text.replace(/(\r\n|\n|\r)/gm, "");
    if (prompt.includes("Prompt:")) {
        prompt = prompt.replace("Prompt:", "")
    }
    if (numImages === 1) {
        numImages = ""
    }
    try {
        await msg.channel.send("^" + command + numImages + " " + prompt)
    } catch (error) {
        console.log("Error getting completion: " + error);
        await msg.reply("Error getting completion. Try again later.");
    }
}