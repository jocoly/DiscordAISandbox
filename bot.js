import {} from "dotenv/config";
import {Client, Events, GatewayIntentBits} from 'discord.js';
import {draw} from "./commands/draw.js";
import {video} from "./commands/video.js";
import {img2img} from "./commands/img2img.js";
import {xlvid} from "./commands/xlvid.js";
import {upscale} from "./commands/upscale.js";
import {realisticVision} from "./commands/realisticVision.js";
import {openjourney} from "./commands/openjourney.js";
import {dreamShaper} from "./commands/dreamShaper.js";
import {anything} from "./commands/anything.js";
import {dreamlikePhotoreal} from "./commands/dreamlikePhotoreal.js";

export const client = new Client({intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent,],});

export const queue = [];

let CONTAIN_BOT = false;
if (process.env.CONTAIN_BOT === 'true') {
    CONTAIN_BOT = true;
}
let STABLE_DIFFUSION = false;
if (process.env.STABLE_DIFFUSION === 'true') {
    STABLE_DIFFUSION = true;
}
let TEXT_TO_VIDEO = false;
if (process.env.TEXT_TO_VIDEO === 'true') {
    TEXT_TO_VIDEO = true;
}
let IMAGE_TO_IMAGE = false;
if (process.env.IMAGE_TO_IMAGE === 'true') {
    IMAGE_TO_IMAGE = true;
}
let XL_VIDEO = false;
if (process.env.XL_VIDEO === 'true') {
    XL_VIDEO = true;
}
let UPSCALE = false;
if (process.env.UPSCALE === 'true') {
    UPSCALE = true;
}
let REALISTIC = false;
if (process.env.REALISTIC === 'true') {
    REALISTIC = true;
}
let OPENJOURNEY = false;
if (process.env.OPENJOURNEY === 'true') {
    OPENJOURNEY = true;
}
let DREAM_SHAPER = false;
if (process.env.REALISTIC === 'true') {
    DREAM_SHAPER = true;
}
let ANYTHING_V3 = false;
if (process.env.ANYTHING_V3 === 'true') {
    ANYTHING_V3 = true;
}
let DREAMLIKE_PHOTOREAL = false;
if (process.env.DREAMLIKE_PHOTOREAL === 'true') {
    DREAMLIKE_PHOTOREAL = true;
}

let DISCORD_CHANNEL_ID = process.env.DISCORD_CHANNEL_ID;

client.once(Events.ClientReady, c => {
    console.log(`Logged in as ${client.user.tag}.`);
});

client.on(Events.MessageCreate, async msg => {
    if (msg.author.id === client.user.id) return;
    if (CONTAIN_BOT && msg.channel.id !== DISCORD_CHANNEL_ID) return;

    if (msg.content.includes("!test")) {
        await msg.reply("Hello world!")
    }

    if (msg.content.includes("!draw") && STABLE_DIFFUSION) {
        await draw(msg);
    }

    if ((msg.content.includes("!realistic") || msg.content.includes("!rv")) && REALISTIC) {
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

    if (msg.content.includes('!img2img') && IMAGE_TO_IMAGE) {
        await img2img(msg);
    }

    if (msg.content.includes('!xlvid') && XL_VIDEO) {
        await xlvid(msg);
    }

    if (msg.content.includes('!upscale') && UPSCALE) {
        await upscale(msg);
    }
 });
await client.login(process.env.DISCORD_TOKEN)