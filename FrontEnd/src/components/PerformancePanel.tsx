import type { PerformanceAggregate, PerformanceSample } from "../types";


interface PerformancePanelProps {
  latestSample: PerformanceSample | null;
  aggregate: PerformanceAggregate;
  benchmarkMode: boolean;
}


const latestMetricLabels = [
  ["time_to_first_visible_char_ms", "首字符可见"],
  ["time_to_full_visible_answer_ms", "完整回答可见"],
  ["server_retrieval_ms", "后端检索"],
  ["server_time_to_first_token_ms", "后端首 token"],
  ["server_total_ms", "后端总耗时"],
] as const;


export default function PerformancePanel({
  latestSample,
  aggregate,
  benchmarkMode,
}: PerformancePanelProps) {
  if (!latestSample && !aggregate.sample_count) {
    return null;
  }

  return (
    <section className="panel performance-panel" aria-label="性能指标">
      <div className="performance-panel-header">
        <div>
          <p className="eyebrow">Latency</p>
          <h2 className="performance-title">流式回答性能</h2>
        </div>
        <span className="performance-count">
          已记录 {benchmarkMode ? `${aggregate.sample_count} 次完成样本` : "最近一次请求"}
        </span>
      </div>

      {latestSample ? (
        <div className="performance-section">
          <div className="performance-section-header">
            <h3>最近一次</h3>
            <span>{latestSample.status === "done" ? "已完成" : latestSample.status === "aborted" ? "已暂停" : "失败"}</span>
          </div>
          <div className="performance-grid">
            {latestMetricLabels.map(([metricKey, label]) => (
              <article className="performance-card" key={metricKey}>
                <span>{label}</span>
                <strong>{formatMs(latestSample[metricKey])}</strong>
              </article>
            ))}
          </div>
        </div>
      ) : null}

      {benchmarkMode && aggregate.sample_count ? (
        <div className="performance-section">
          <div className="performance-section-header">
            <h3>聚合结果</h3>
            <span>avg / p50 / p95 / min / max</span>
          </div>
          <div className="performance-aggregate-list">
            {latestMetricLabels.map(([metricKey, label]) => {
              const metric = aggregate.metrics[metricKey];
              if (!metric) {
                return null;
              }

              return (
                <div className="performance-aggregate-row" key={metricKey}>
                  <span>{label}</span>
                  <strong>
                    {formatMs(metric.avg)} / {formatMs(metric.p50)} / {formatMs(metric.p95)} / {formatMs(metric.min)} / {formatMs(metric.max)}
                  </strong>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}


function formatMs(value: number | null | undefined): string {
  return typeof value === "number" ? `${value.toFixed(2)} ms` : "未采集";
}
