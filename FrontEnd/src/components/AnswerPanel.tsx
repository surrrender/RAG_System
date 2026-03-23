import type { QAResponse } from "../types";


interface AnswerPanelProps {
  result: QAResponse | null;
  loading: boolean;
}


export default function AnswerPanel({ result, loading }: AnswerPanelProps) {
  if (loading) {
    return (
      <section className="panel answer-panel answer-loading">
        <div className="panel-header">
          <p className="eyebrow">回答区</p>
          <h2>正在生成答案</h2>
        </div>
        <p className="answer-placeholder">正在检索相关文档并生成回答，请稍候。</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel answer-panel answer-empty">
        <div className="panel-header">
          <p className="eyebrow">回答区</p>
          <h2>等待提问</h2>
        </div>
        <p className="answer-placeholder">提交一个问题后，这里会展示回答、模型信息和引用来源。</p>
      </section>
    );
  }

  return (
    <section className="panel answer-panel">
      <div className="panel-header">
        <p className="eyebrow">回答区</p>
        <h2>生成结果</h2>
      </div>
      <div className="answer-meta">
        <span>模型：{result.model}</span>
        <span>命中文档：{result.retrieval_count}</span>
      </div>
      <div className="answer-body">
        {result.answer.split("\n").map((paragraph, index) => (
          <p key={`${paragraph}-${index}`}>{paragraph || "\u00A0"}</p>
        ))}
      </div>
    </section>
  );
}
