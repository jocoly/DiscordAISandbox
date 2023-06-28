export function getNumImages(msg) {
    const content = msg.content.split(" ")[0].replace(/\D/g, '');
    if (content === '') return 1;
    else return content;
}