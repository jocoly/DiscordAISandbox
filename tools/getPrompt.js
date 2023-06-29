export function getPrompt(msg) {
    if (msg.content.startsWith("!")) {
            return msg.content.split(" ").slice(1).join(' ');
    } else return msg.content;
}