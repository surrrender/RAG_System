import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";


const trackedMetricKeys = [
  "time_to_first_visible_char_ms",
  "time_to_full_visible_answer_ms",
  "server_retrieval_ms",
  "server_time_to_first_token_ms",
  "server_total_ms",
];
const defaultWaitTimeoutMs = Number(process.env.QA_BENCHMARK_TIMEOUT_MS ?? 1200000);
const rawDataConverterPath = fileURLToPath(new URL("./convert-latency-json-to-raw-data.mjs", import.meta.url));


async function main() {
  const questionsFilePath = resolveQuestionsFilePath(process.argv[2]);
  const outputFilePath = resolveOutputFilePath(process.argv[3]);
  const appUrl = process.env.QA_BENCHMARK_APP_URL?.trim() || "http://localhost:5173/?benchmark=1";
  const questions = loadQuestions(questionsFilePath);

  if (!questions.length) {
    throw new Error(`No questions found in ${questionsFilePath}.`);
  }

  const browser = await chromium.launch({
    headless: process.env.QA_BENCHMARK_HEADLESS !== "0",
  });

  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(defaultWaitTimeoutMs);

    const samples = [];
    for (let index = 0; index < questions.length; index += 1) {
      await page.goto(appUrl, { waitUntil: "networkidle" });

      const textarea = page.getByLabel("问题");
      const submitButton = page.getByRole("button", { name: "发送问题" });
      await textarea.waitFor({ state: "visible" });
      await submitButton.waitFor({ state: "visible" });

      await textarea.fill(questions[index]);
      await submitButton.click();

      try {
        await page.waitForFunction(
          () => {
            const metrics = window.__qaMetrics__;
            return Boolean(metrics?.latestSample?.status === "done");
          },
          { timeout: defaultWaitTimeoutMs },
        );
      } catch (error) {
        throw await buildWaitTimeoutError(page, questions[index], index, error);
      }

      const latestSample = await page.evaluate(() => window.__qaMetrics__?.latestSample ?? null);
      if (!latestSample) {
        throw new Error(`No latest sample was recorded for question #${index + 1}.`);
      }
      if (latestSample.status !== "done") {
        throw new Error(
          [
            `Question #${index + 1} did not finish successfully.`,
            `question: ${questions[index]}`,
            `status: ${latestSample.status}`,
            latestSample.error_message ? `error: ${latestSample.error_message}` : null,
          ]
            .filter(Boolean)
            .join("\n"),
        );
      }

      samples.push(
        Object.fromEntries(trackedMetricKeys.map((metricKey) => [metricKey, latestSample[metricKey] ?? null])),
      );
      globalThis.__benchmarkPartialSamples = samples;

      await submitButton.waitFor({ state: "visible" });
    }

    writeSamples(outputFilePath, samples);
    updateDashboardFromSamplesFile(outputFilePath);
    process.stdout.write(`Saved ${samples.length} benchmark samples to ${outputFilePath}\n`);
  } catch (error) {
    if (typeof globalThis.__benchmarkPartialSamples !== "undefined") {
      writeSamples(outputFilePath, globalThis.__benchmarkPartialSamples);
      try {
        updateDashboardFromSamplesFile(outputFilePath);
      } catch (postProcessError) {
        process.stderr.write(
          `${postProcessError instanceof Error ? postProcessError.message : String(postProcessError)}\n`,
        );
      }
    }
    throw error;
  } finally {
    await browser.close();
  }
}


function resolveQuestionsFilePath(cliPath) {
  const candidate = cliPath || process.env.QA_BENCHMARK_QUESTIONS_FILE;
  if (!candidate) {
    throw new Error("Usage: npm run benchmark:file -- <questions-file> [output-file]");
  }
  return path.resolve(process.cwd(), candidate);
}


function resolveOutputFilePath(cliPath) {
  const candidate = cliPath || process.env.QA_BENCHMARK_OUTPUT_FILE || "benchmark-results/question-latency-results.json";
  return path.resolve(process.cwd(), candidate);
}


function loadQuestions(filePath) {
  const raw = fs.readFileSync(filePath, "utf8").trim();
  const extension = path.extname(filePath).toLowerCase();

  if (!raw) {
    return [];
  }

  if (extension === ".json") {
    const data = JSON.parse(raw);
    if (!Array.isArray(data) || !data.every((item) => typeof item === "string")) {
      throw new Error("JSON question files must contain an array of strings.");
    }
    return data.map((item) => item.trim()).filter(Boolean);
  }

  if (extension === ".jsonl") {
    return raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const parsed = JSON.parse(line);
        if (typeof parsed === "string") {
          return parsed.trim();
        }
        if (parsed && typeof parsed.question === "string") {
          return parsed.question.trim();
        }
        throw new Error("Each JSONL line must be a string or an object with a question field.");
      })
      .filter(Boolean);
  }

  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => Boolean(line) && !line.startsWith("#"));
}


main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});


async function buildWaitTimeoutError(page, question, index, originalError) {
  const diagnostics = await page.evaluate(() => {
    const metrics = window.__qaMetrics__;
    const errorNode = document.querySelector(".message-error .markdown-body, .message-error .message-error-tip");
    const validationNode = document.querySelector(".composer-error");
    const assistantNodes = document.querySelectorAll(".message-assistant .markdown-body");
    const latestAssistantNode = assistantNodes.length > 0 ? assistantNodes[assistantNodes.length - 1] : null;

    return {
      latestSample: metrics?.latestSample ?? null,
      sampleCount: metrics?.samples.length ?? 0,
      errorText: errorNode?.textContent?.trim() ?? null,
      validationText: validationNode?.textContent?.trim() ?? null,
      assistantPreview: latestAssistantNode?.textContent?.trim().slice(0, 200) ?? null,
      buttonText:
        document.querySelector('button[type="submit"], button[type="button"]')?.textContent?.trim() ?? null,
    };
  });

  return new Error(
    [
      `Timed out after ${defaultWaitTimeoutMs}ms while waiting for question #${index + 1} to finish.`,
      `question: ${question}`,
      `samples recorded: ${diagnostics.sampleCount}`,
      diagnostics.latestSample ? `latest sample status: ${diagnostics.latestSample.status}` : "latest sample status: none",
      diagnostics.latestSample?.error_message ? `latest sample error: ${diagnostics.latestSample.error_message}` : null,
      diagnostics.errorText ? `page error: ${diagnostics.errorText}` : null,
      diagnostics.validationText ? `validation error: ${diagnostics.validationText}` : null,
      diagnostics.assistantPreview ? `assistant preview: ${diagnostics.assistantPreview}` : null,
      diagnostics.buttonText ? `current button text: ${diagnostics.buttonText}` : null,
      originalError instanceof Error ? `playwright: ${originalError.message}` : null,
    ]
      .filter(Boolean)
      .join("\n"),
  );
}


function writeSamples(outputFilePath, samples) {
  fs.mkdirSync(path.dirname(outputFilePath), { recursive: true });
  fs.writeFileSync(outputFilePath, JSON.stringify(samples, null, 2), "utf8");
}


function updateDashboardFromSamplesFile(outputFilePath) {
  const result = spawnSync(process.execPath, [rawDataConverterPath, outputFilePath], {
    stdio: "inherit",
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(`Failed to update latency dashboard from ${outputFilePath}.`);
  }
}
