export function getPrompt(msg) {
    return msg.content.split(" ").slice(1).join(' ');
}