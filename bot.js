import {} from "dotenv/config";
import {Client, Events, GatewayIntentBits} from 'discord.js';
import {Configuration, OpenAIApi} from 'openai';
import {ask} from "./commands/ask.js";
import {chat} from "./commands/chat.js";
import {prompt} from "./commands/chat.js";
import {random} from "./commands/random.js";
import {draw} from "./commands/draw.js";
import {img2img} from "./commands/img2img.js";
import {upscale} from "./commands/upscale.js";
import {video} from "./commands/video.js";
import {animov} from "./commands/animov.js";
import {xlvid} from "./commands/xlvid.js";
import {audio} from "./commands/audio.js";
import {speech} from "./commands/speech.js";
import {caption} from "./commands/caption.js";
import {realisticVision} from "./commands/realisticVision.js";
import {openjourney} from "./commands/openjourney.js";
import {dreamShaper} from "./commands/dreamShaper.js";
import {anything} from "./commands/anything.js";
import {dreamlikePhotoreal} from "./commands/dreamlikePhotoreal.js";
import {waifuDiffusion} from "./commands/waifuDiffusion.js";
import {vox2} from "./commands/vox2.js";

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
let ANIMOV = process.env.ANIMOV_512X === 'true';
let XL_VIDEO = process.env.XL_VIDEO === 'true';
let UPSCALE = process.env.UPSCALE === 'true';
let REALISTIC_VISION = process.env.REALISTIC_VISION === 'true';
let OPENJOURNEY = process.env.OPENJOURNEY === 'true';
let DREAM_SHAPER = process.env.DREAM_SHAPER === 'true';
let ANYTHING_V3 = process.env.ANYTHING_V3 === 'true';
let DREAMLIKE_PHOTOREAL = process.env.DREAMLIKE_PHOTOREAL === 'true';
let WAIFU_DIFFUSION = process.env.WAIFU_DIFFUSION === 'true';
let VOX2 = process.env.VOX2 === 'true';

let DISCORD_CHANNEL_ID = process.env.DISCORD_CHANNEL_ID;

client.once(Events.ClientReady, c => {
    console.log(`Logged in as ${client.user.tag}.`);
});

const configuration = new Configuration ({apiKey: process.env.OPENAI_TOKEN,});
export const openai = new OpenAIApi(configuration);


client.on(Events.MessageCreate, async msg => {
    let msgContent = msg.content;
    if (msg.author.id === client.user.id && msg.content.substring(0,1) !== ('^')) return;
    if (CONTAIN_BOT && msg.channel.id !== DISCORD_CHANNEL_ID) return;
    let isReply, refMsg, isCommand, isMention;
    try {
        isCommand = Array.from(msg.content)[0] === '!';
        // people like to tag their friends in the bot's gpt replies sometimes
        // if a mention is the first part of a reply to the bot, the bot will not process it for gpt response
        isMention = Array.from(msg.content)[0] === '<';
        refMsg = await msg.fetchReference()
        isReply = true;
    } catch (error) {
        isReply = false;
    }
    if (msgContent.substring(0,1) === '^') {
        msgContent = msgContent.replace('^', '');
    }
    if (isReply && !isCommand && !isMention && refMsg.author.id === client.user.id) {
        await chat(msg)
    }

    if (msgContent.substring(0, 5) === ("!test")) {
        await msg.reply("Hello world!")
    }

    if (msgContent.substring(0, 4) === ("!ask") && ASK) {
        await ask(msg);
    }

    if (msgContent.substring(0, 5) === ("!chat") && CHAT) {
        await chat(msg);
    }

    if (msgContent.substring(0, 5) === ("!draw") && STABLE_DIFFUSION) {
        await draw(msg);
    }

    if ((msgContent.substring(0, 10) === ("!realistic") || msgContent.substring(0, 3) === ("!rv")) && REALISTIC_VISION) {
        await realisticVision(msg);
    }

    if ((msgContent.substring(0, 12) === ("!openjourney") || msgContent.substring(0, 3) === ("!oj")) && OPENJOURNEY) {
        await openjourney(msg);
    }

    if ((msgContent.substring(0, 12) === ("!dreamshaper") || msgContent.substring(0, 3) === ("!ds")) && DREAM_SHAPER) {
        await dreamShaper(msg);
    }

    if (msgContent.substring(0, 9) === ("!anything") && ANYTHING_V3) {
        await anything(msg);
    }

    if (msgContent.substring(0, 7) === ('!prompt')) {
        await prompt(msg);
    } else if ((msgContent.substring(0, 10) === ("!photoreal") || msgContent.substring(0, 3) === ("!pr")) && DREAMLIKE_PHOTOREAL) {
        await dreamlikePhotoreal(msg);
    }

    if ((msgContent.substring(0, 6) === ("!waifu") || msgContent.substring(0, 3) === ("!wd")) && WAIFU_DIFFUSION) {
        await waifuDiffusion(msg);
    }

    if (msgContent.substring(0, 4) === ("!vox") && VOX2) {
        await vox2(msg);
    }

    if (msgContent.substring(0, 6) === ('!video') && TEXT_TO_VIDEO) {
        await video(msg);
    }

    if (msgContent.substring(0, 6) === ('!audio') && TEXT_TO_AUDIO) {
        await audio(msg);
    }

    if (msgContent.substring(0, 7) === ('!speech') && TEXT_TO_SPEECH) {
        await speech(msg);
    }

    if (msgContent.substring(0, 8) === ('!img2img') && IMAGE_TO_IMAGE) {
        await img2img(msg);
    }

    if (msgContent.substring(0, 7) === ('!animov') && ANIMOV) {
        await animov(msg);
    }

    if (msgContent.substring(0, 6) === ('!xlvid') && XL_VIDEO) {
        await xlvid(msg);
    }

    if (msgContent.substring(0, 8) === ('!upscale') && UPSCALE) {
        await upscale(msg);
    }

    if (msgContent.substring(0, 8) === ('!caption') && CAPTION) {
        await caption(msg);
    }

    if (msgContent.substring(0, 7) === ('!random')) {
        await random(msg);
    }
 });
await client.login(process.env.DISCORD_TOKEN)