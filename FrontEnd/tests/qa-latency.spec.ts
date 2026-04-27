import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";


test("records visible latency samples and exports aggregate results", async ({ page }) => {
  const iterations = Number(process.env.QA_BENCHMARK_ITERATIONS ?? 3);
  const question = "小程序 App 生命周期是什么？";

  await page.goto("/?benchmark=1");

  for (let index = 0; index < iterations; index += 1) {
    const textarea = page.getByLabel("问题");
    await textarea.fill(question);
    await page.getByRole("button", { name: "发送问题" }).click();

    await expect(page.getByText("这是用于性能基准的稳定流式回答。").last()).toBeVisible();
    await page.waitForFunction(
      (expectedCount) => window.__qaMetrics__?.samples.length === expectedCount,
      index + 1,
    );
  }

  const metrics = await page.evaluate(() => window.__qaMetrics__);
  expect(metrics?.aggregate.sample_count).toBe(iterations);
  expect(metrics?.latestSample?.time_to_first_visible_char_ms).not.toBeNull();
  expect(metrics?.latestSample?.time_to_full_visible_answer_ms).not.toBeNull();

  const outputDirectory = path.join(process.cwd(), "benchmark-results");
  fs.mkdirSync(outputDirectory, { recursive: true });

  const jsonPath = path.join(outputDirectory, "qa-latency.json");
  fs.writeFileSync(jsonPath, JSON.stringify(metrics, null, 2), "utf8");

  const csvRows = [
    [
      "request_id",
      "question",
      "status",
      "time_to_first_visible_char_ms",
      "time_to_full_visible_answer_ms",
      "server_retrieval_ms",
      "server_time_to_first_token_ms",
      "server_total_ms",
    ].join(","),
    ...(metrics?.samples ?? []).map((sample) =>
      [
        sample.request_id,
        escapeCsv(sample.question),
        sample.status,
        sample.time_to_first_visible_char_ms ?? "",
        sample.time_to_full_visible_answer_ms ?? "",
        sample.server_retrieval_ms ?? "",
        sample.server_time_to_first_token_ms ?? "",
        sample.server_total_ms ?? "",
      ].join(","),
    ),
  ];
  fs.writeFileSync(path.join(outputDirectory, "qa-latency.csv"), csvRows.join("\n"), "utf8");
});


function escapeCsv(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}
