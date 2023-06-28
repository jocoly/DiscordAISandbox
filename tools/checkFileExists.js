import fs from "fs";

export async function checkFileExists(filePath) {
    try {
        await fs.promises.stat(filePath);
        return true;
    } catch (error) {
        return false;
    }
}