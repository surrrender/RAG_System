import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";


const requiredKeys = [
  "time_to_first_visible_char_ms",
  "time_to_full_visible_answer_ms",
  "server_retrieval_ms",
];
const optionalStageKeys = [
  "server_embed_ms",
  "server_vector_search_ms",
  "server_rerank_ms",
  "server_prompt_build_ms",
];

const defaultHtmlPath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../benchmark-results/question-latency-dashboard.html",
);


function main() {
  const inputArg = process.argv[2];
  const outputArg = process.argv[3];

  if (!inputArg) {
    throw new Error(
      "Usage: npm run benchmark:rawdata -- <input-json-file> [output-file]",
    );
  }

  const inputPath = path.resolve(process.cwd(), inputArg);
  const outputPath = outputArg
    ? path.resolve(process.cwd(), outputArg)
    : defaultHtmlPath;
  const raw = fs.readFileSync(inputPath, "utf8");
  const parsed = JSON.parse(raw);
  const normalized = normalizeSamples(parsed);

  if (outputPath) {
    if (/\.html?$/i.test(outputPath)) {
      const html = fs.readFileSync(outputPath, "utf8");
      const updatedHtml = replaceRawDataBlock(html, normalized);
      fs.writeFileSync(outputPath, updatedHtml, "utf8");
      process.stdout.write(`Updated rawData block in ${outputPath}\n`);
      return;
    }

    const output = buildRawDataSnippet(normalized);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, output, "utf8");
    process.stdout.write(`Saved rawData snippet to ${outputPath}\n`);
    return;
  }

  const output = buildRawDataSnippet(normalized);
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
      server_embed_ms: toOptionalNumber(item.server_embed_ms),
      server_vector_search_ms: toOptionalNumber(item.server_vector_search_ms),
      server_rerank_ms: toOptionalNumber(item.server_rerank_ms),
      server_prompt_build_ms: toOptionalNumber(item.server_prompt_build_ms),
    };
  });
}


function buildRawDataSnippet(samples, indent = "") {
  const lines = samples.map(
    (item) => `${indent}  ${formatRawDataItem(item)}`,
  );

  return [`${indent}const rawData = [`, ...lines.map((line, index) => `${line}${index === lines.length - 1 ? "" : ","}`), `${indent}];`].join("\n");
}


function replaceRawDataBlock(html, samples) {
  const startMarker = "const rawData = [";
  const startIndex = html.indexOf(startMarker);

  if (startIndex === -1) {
    throw new Error("Could not find `const rawData = [` in the target HTML file.");
  }

  const arrayContentStart = startIndex + startMarker.length;
  const arrayEndIndex = findMatchingClosingBracket(html, arrayContentStart - 1);

  if (arrayEndIndex === -1) {
    throw new Error("Could not find the matching closing `]` for rawData.");
  }

  const lineStart = html.lastIndexOf("\n", startIndex) + 1;
  const indent = html.slice(lineStart, startIndex);

  const newDataContent = buildRawDataArrayItems(samples, indent + "  ");

  return (
    html.slice(0, arrayContentStart) +
    "\n" +
    newDataContent +
    "\n" +
    indent +
    html.slice(arrayEndIndex)
  );
}

function findMatchingClosingBracket(text, openBracketIndex) {
  let depth = 0;
  let inString = null;
  let escaped = false;

  for (let i = openBracketIndex; i < text.length; i++) {
    const char = text[i];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === inString) {
        inString = null;
      }
      continue;
    }

    if (char === '"' || char === "'" || char === "`") {
      inString = char;
      continue;
    }

    if (char === "[") depth++;
    if (char === "]") depth--;

    if (depth === 0) {
      return i;
    }
  }

  return -1;
}

function buildRawDataArrayItems(samples, itemIndent) {
  return samples
    .map((sample) => {
      return itemIndent + formatRawDataItem(sample);
    })
    .join(",\n");
}


function formatRawDataItem(sample) {
  return JSON.stringify(sample).replace(/"([^"]+)":/g, "$1:");
}


function toOptionalNumber(value) {
  if (value === null || typeof value === "undefined") {
    return null;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }
  return value;
}

main();
