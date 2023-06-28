import * as path from "path";
import * as fs from "fs";
import fetch from "node-fetch";

const REQUEST_TIMEOUT_SEC = 120000
export async function callBackendPipeline(text_prompt, pipeline, num) {
    const start_time = new Date();
    const backendUrl = process.env.BACKEND_ADDRESS + ":" + process.env.PORT
    const response = await Promise.race([
        fetch("http://" + backendUrl + "/process", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text_prompt,
                pipeline,
                num
            })
        }).then((response) => {
            if (!response.ok) {
                throw Error(response.statusText);
            }
            return response;
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), REQUEST_TIMEOUT_SEC))
    ]);
    const results = [];
    const jsonResponse = await response.json();

    for (const file of jsonResponse.generation) {
        results.push(file);
    }

    const end_time = new Date();
    console.log(`Query took ${end_time - start_time} ms`);
    return results;
}