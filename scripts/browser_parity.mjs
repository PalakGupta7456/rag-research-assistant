// Proves the browser model (Transformers.js, quantised ONNX) produces the same
// embeddings as the Python model used to build the index. Reads Python vectors
// from stdin-produced JSON and prints the per-sentence cosine agreement.
import { pipeline } from "@huggingface/transformers";
import { readFileSync } from "node:fs";

const { sentences, vectors } = JSON.parse(readFileSync(process.argv[2], "utf8"));
const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8" });
const out = (await embed(sentences, { pooling: "mean", normalize: true })).tolist();
const cos = out.map((v, i) => v.reduce((a, x, j) => a + x * vectors[i][j], 0));
console.log("cosine(python, browser):", cos.map((c) => c.toFixed(4)).join(" "));
if (Math.min(...cos) < 0.97) { console.error("Browser and Python embeddings disagree"); process.exit(1); }
