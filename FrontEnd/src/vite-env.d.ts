/// <reference types="vite/client" />

import type { PerformanceAggregate, PerformanceSample } from "./types";

declare global {
  interface Window {
    __qaMetrics__?: {
      benchmarkMode: boolean;
      latestSample: PerformanceSample | null;
      samples: PerformanceSample[];
      aggregate: PerformanceAggregate;
    };
  }
}
