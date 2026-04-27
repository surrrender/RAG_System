import type {
  AggregatedMetric,
  PerformanceAggregate,
  PerformanceMetricKey,
  PerformanceSample,
} from "../types";


const trackedMetricKeys: PerformanceMetricKey[] = [
  "time_to_first_visible_char_ms",
  "time_to_full_visible_answer_ms",
  "server_retrieval_ms",
  "server_time_to_first_token_ms",
  "server_total_ms",
];


export function computeDelta(start: number, end: number | null): number | null {
  return typeof end === "number" ? roundMs(end - start) : null;
}


export function computeServerDelta(start: number | null, end: number | null): number | null {
  return typeof start === "number" && typeof end === "number" ? roundMs(end - start) : null;
}


export function roundMs(value: number): number {
  return Math.round(value * 100) / 100;
}


export function buildPerformanceAggregate(samples: PerformanceSample[]): PerformanceAggregate {
  const completedSamples = samples.filter((sample) => sample.status === "done");
  const metrics = Object.fromEntries(
    trackedMetricKeys.flatMap((metricKey) => {
      const values = completedSamples
        .map((sample) => sample[metricKey])
        .filter((value): value is number => typeof value === "number");

      return values.length ? [[metricKey, summarize(values)]] : [];
    }),
  ) as Partial<Record<PerformanceMetricKey, AggregatedMetric>>;

  return {
    sample_count: completedSamples.length,
    metrics,
  };
}


function summarize(values: number[]): AggregatedMetric {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    avg: roundMs(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    min: roundMs(sorted[0]),
    max: roundMs(sorted[sorted.length - 1]),
  };
}


function percentile(sortedValues: number[], ratio: number): number {
  const index = Math.min(sortedValues.length - 1, Math.max(0, Math.ceil(sortedValues.length * ratio) - 1));
  return roundMs(sortedValues[index]);
}
