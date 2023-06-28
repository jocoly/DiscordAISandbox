import {} from "dotenv/config";
import {Client, Events, GatewayIntentBits} from 'discord.js';
import {draw} from "./commands/draw.js";
import {video} from "./commands/video.js";

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

    if (msg.content.includes('!video') && TEXT_TO_VIDEO) {
        await video(msg);
    }
});
await client.login(process.env.DISCORD_TOKEN)