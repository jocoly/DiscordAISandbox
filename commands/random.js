import {client, openai} from "../bot.js";
import {getPrompt} from "../tools/getPrompt.js";
import {getNumImages} from "../tools/getNumImages.js";

const maxNumImgs = process.env.MAX_NUM_IMAGES;
export const imageModelCommands = [
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
];

export async function getTopic(msg) {
    let userTopic = getPrompt(msg).replace(/\s+/g, '');
    if (userTopic === "") {
        const completion = await openai.createCompletion({
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
    const topic = await getTopic(msg) + ".";
    let numImages = await getNumImages(msg);
    console.log(topic);
    const command = imageModelCommands[Math.floor(Math.random()*imageModelCommands.length)]
    const completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "You will now act as an image description generator. I will describe an topic to you, and you will add details create a description that could be used for image-generation. The topic I want to generate an image for is: " + topic,
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: Number(process.env.CHAT_TEMPERATURE)
    })
    let prompt = completion.data.choices[0].text.replace(/(\r\n|\n|\r)/gm, "");
    if (prompt.includes("Description:")) {
        prompt = prompt.replace("Description:", "")
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

export async function randomGen() {
    const topic = await (await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "Name a subject or scene that would make for an interesting image: ",
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: 1.5
    })).data.choices[0].text
    console.log("RandomGen topic: " + topic)
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
    const discordChannelId = await client.channels.fetch(process.env.DISCORD_CHANNEL_ID);
    const completion = await openai.createCompletion({
        model: process.env.CHAT_MODEL,
        prompt: "You will now act as an image description generator. I will describe an topic to you, and you will add details create a description that could be used for image-generation. The topic I want to generate an image for is: " + topic,
        max_tokens: Number(process.env.CHAT_PROMPT_MAX_TOKENS),
        temperature: Number(process.env.CHAT_TEMPERATURE)
    })
    let prompt = completion.data.choices[0].text.replace(/(\r\n|\n|\r)/gm, "");
    if (prompt.includes("Description:")) {
        prompt = prompt.replace("Description:", "")
    }
    try {
        await discordChannelId.send("^" + command + maxNumImgs + " " + prompt)
    } catch (error) {
        console.log("Error getting completion for random image gen: " + error);
    }
}
