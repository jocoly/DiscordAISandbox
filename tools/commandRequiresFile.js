export async function commandRequiresFile(msg) {
    return !!msg.content.includes("!upscale") || msg.content.includes("!img2img") || msg.content.includes("!caption");
}