import fs from "node:fs";
import path from "node:path";
import process from "node:process";


const requiredKeys = [
  "time_to_first_visible_char_ms",
  "time_to_full_visible_answer_ms",
  "server_retrieval_ms",
];


function main() {
  const inputArg = process.argv[2];
  const outputArg = process.argv[3];

  if (!inputArg) {
    throw new Error(
      "Usage: npm run benchmark:rawdata -- <input-json-file> [output-file]",
    );
  }

  const inputPath = path.resolve(process.cwd(), inputArg);
  const outputPath = outputArg ? path.resolve(process.cwd(), outputArg) : null;
  const raw = fs.readFileSync(inputPath, "utf8");
  const parsed = JSON.parse(raw);
  const normalized = normalizeSamples(parsed);
  const output = buildRawDataSnippet(normalized);

  if (outputPath) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, output, "utf8");
    process.stdout.write(`Saved rawData snippet to ${outputPath}\n`);
    return;
  }

  process.stdout.write(`${output}\n`);
}


function normalizeSamples(parsed) {
  if (!Array.isArray(parsed)) {
    throw new Error("Input JSON must be an array.");
  }

  return parsed.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Item #${index + 1} must be an object.`);
    }

    for (const key of requiredKeys) {
      if (!(key in item)) {
        throw new Error(`Item #${index + 1} is missing required field: ${key}`);
      }
      if (typeof item[key] !== "number" || Number.isNaN(item[key])) {
        throw new Error(`Item #${index + 1} field ${key} must be a valid number.`);
      }
    }

    return {
      time_to_first_visible_char_ms: item.time_to_first_visible_char_ms,
      time_to_full_visible_answer_ms: item.time_to_full_visible_answer_ms,
      server_retrieval_ms: item.server_retrieval_ms,
    };
  });
}


function buildRawDataSnippet(samples) {
  const lines = samples.map(
    (item) =>
      `  { time_to_first_visible_char_ms: ${item.time_to_first_visible_char_ms}, time_to_full_visible_answer_ms: ${item.time_to_full_visible_answer_ms}, server_retrieval_ms: ${item.server_retrieval_ms} }`,
  );

  return ["const rawData = [", ...lines.map((line, index) => `${line}${index === lines.length - 1 ? "" : ","}`), "];"].join("\n");
}


main();
