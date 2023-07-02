import {} from "dotenv/config";
import {Client, Events, GatewayIntentBits} from 'discord.js';
import {Configuration, OpenAIApi} from 'openai';
import {ask} from "./commands/ask.js";
import {chat} from "./commands/chat.js";
import {draw} from "./commands/draw.js";
import {video} from "./commands/video.js";
import {caption} from "./commands/caption.js";
import {img2img} from "./commands/img2img.js";
import {xlvid} from "./commands/xlvid.js";
import {upscale} from "./commands/upscale.js";
import {realisticVision} from "./commands/realisticVision.js";
import {openjourney} from "./commands/openjourney.js";
import {dreamShaper} from "./commands/dreamShaper.js";
import {anything} from "./commands/anything.js";
import {dreamlikePhotoreal} from "./commands/dreamlikePhotoreal.js";
import {audio} from "./commands/audio.js";
import {speech} from "./commands/speech.js";

export const client = new Client({intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent,],});

export const queue = [];

let CONTAIN_BOT = process.env.CONTAIN_BOT === 'true';
let CHAT = process.env.CHAT === 'true';
let ASK = process.env.ASK === 'true';
let STABLE_DIFFUSION = process.env.STABLE_DIFFUSION === 'true';
let TEXT_TO_VIDEO = process.env.TEXT_TO_VIDEO === 'true';
let TEXT_TO_AUDIO = process.env.TEXT_TO_AUDIO === 'true';
let TEXT_TO_SPEECH = process.env.TEXT_TO_SPEECH === 'true';
let CAPTION = process.env.CAPTION === 'true';
let IMAGE_TO_IMAGE = process.env.IMAGE_TO_IMAGE === 'true';
let XL_VIDEO = process.env.XL_VIDEO === 'true';
let UPSCALE = process.env.UPSCALE === 'true';
let REALISTIC_VISION = process.env.REALISTIC_VISION === 'true';
let OPENJOURNEY = process.env.OPENJOURNEY === 'true';
let DREAM_SHAPER = process.env.DREAM_SHAPER === 'true';
let ANYTHING_V3 = process.env.ANYTHING_V3 === 'true';
let DREAMLIKE_PHOTOREAL = process.env.DREAMLIKE_PHOTOREAL === 'true';

let DISCORD_CHANNEL_ID = process.env.DISCORD_CHANNEL_ID;

client.once(Events.ClientReady, c => {
    console.log(`Logged in as ${client.user.tag}.`);
});

const configuration = new Configuration ({apiKey: process.env.OPENAI_TOKEN,});
export const openai = new OpenAIApi(configuration);


client.on(Events.MessageCreate, async msg => {
    if (msg.content.includes('!^chat')) await chat(msg); //bot can see his own messages for auto chat
    if (msg.author.id === client.user.id) return;
    if (CONTAIN_BOT && msg.channel.id !== DISCORD_CHANNEL_ID) return;
    let isReply, refMsg, isCommand, isMention;
    try {
        isCommand = Array.from(msg.content)[0] === '!';
        isMention = Array.from(msg.content)[0] === '<';
        refMsg = await msg.fetchReference()
        isReply = true;
    } catch (error) {
        isReply = false;
    }

    if (isReply && !isCommand && !isMention && refMsg.author.id === client.user.id) {
        await chat(msg)
    }

    if (msg.content.includes("!test")) {
        await msg.reply("Hello world!")
    }

    if (msg.content.includes("!ask") && ASK) {
        await ask(msg);
    }

    if (msg.content.includes("!chat") && CHAT) {
        await chat(msg);
    }

    if (msg.content.includes("!draw") && STABLE_DIFFUSION) {
        await draw(msg);
    }

    if ((msg.content.includes("!realistic") || msg.content.includes("!rv")) && REALISTIC_VISION) {
        await realisticVision(msg);
    }

    if ((msg.content.includes("!openjourney") || msg.content.includes("!oj")) && OPENJOURNEY) {
        await openjourney(msg);
    }

    if ((msg.content.includes("!dreamshaper") || msg.content.includes("!ds")) && DREAM_SHAPER) {
        await dreamShaper(msg);
    }

    if (msg.content.includes("!anything") && ANYTHING_V3) {
        await anything(msg);
    }

    if ((msg.content.includes("!photoreal") || msg.content.includes("!pr")) && DREAMLIKE_PHOTOREAL) {
        await dreamlikePhotoreal(msg);
    }

    if (msg.content.includes('!video') && TEXT_TO_VIDEO) {
        await video(msg);
    }

    if (msg.content.includes('!audio') && TEXT_TO_AUDIO) {
        await audio(msg);
    }

    if (msg.content.includes('!speech') && TEXT_TO_SPEECH) {
        await speech(msg);
    }

    if (msg.content.includes('!img2img') && IMAGE_TO_IMAGE) {
        await img2img(msg);
    }

    if (msg.content.includes('!xlvid') && XL_VIDEO) {
        await xlvid(msg);
    }

    if (msg.content.includes('!upscale') && UPSCALE) {
        await upscale(msg);
    }

    if (msg.content.includes('!caption') && CAPTION) {
        await caption(msg);
    }
 });
await client.login(process.env.DISCORD_TOKEN)