export function getCommand(msg) {
    if (msg.content.startsWith("!")) {
            return msg.content.replace(/ .*/,'')
    } else return msg.content;
}