import { useState } from "react";

import { askQuestion } from "./api/client";
import AnswerPanel from "./components/AnswerPanel";
import CitationList from "./components/CitationList";
import QuestionForm from "./components/QuestionForm";
import StatusBanner from "./components/StatusBanner";
import type { QAResponse } from "./types";


const defaultQuestion = "小程序 App 生命周期是什么？";
const defaultTopK = 5;


export default function App() {
  const [question, setQuestion] = useState(defaultQuestion);
  const [topK, setTopK] = useState(defaultTopK);
  const [result, setResult] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleTopKChange = (value: number) => {
    if (Number.isNaN(value)) {
      setTopK(defaultTopK);
      return;
    }

    const clamped = Math.max(1, Math.min(20, Math.trunc(value)));
    setTopK(clamped);
  };

  const handleSubmit = async () => {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion) {
      setValidationError("请输入问题后再提交。");
      return;
    }

    setValidationError(null);
    setErrorMessage(null);
    setLoading(true);

    try {
      const nextResult = await askQuestion({
        question: normalizedQuestion,
        top_k: topK,
      });
      setResult(nextResult);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "请求失败，请检查服务状态。");
    } finally {
      setLoading(false);
    }
  };

  const citations = result?.citations ?? [];

  return (
    <div className="app-shell">
      <div className="app-bg app-bg-top" />
      <div className="app-bg app-bg-bottom" />
      <main className="app-layout">
        <section className="hero">
          <p className="eyebrow">RAG Frontend MVP</p>
          <h1>微信小程序文档问答台</h1>
          <p className="hero-copy">
            连接本地 Ollama 与微信小程序文档索引，让文档检索、答案生成和引用来源都在一个页面中完成。
          </p>
        </section>

        <div className="workspace-grid">
          <div className="left-column">
            <QuestionForm
              question={question}
              topK={topK}
              loading={loading}
              validationError={validationError}
              onQuestionChange={(value) => {
                setQuestion(value);
                if (value.trim()) {
                  setValidationError(null);
                }
              }}
              onTopKChange={handleTopKChange}
              onSubmit={handleSubmit}
            />
            {errorMessage ? <StatusBanner kind="error" message={errorMessage} /> : null}
            {!errorMessage && !result && !loading ? (
              <StatusBanner
                kind="idle"
                message="建议先用默认问题快速验证链路，确认后端 `POST /qa` 已经启动。"
              />
            ) : null}
          </div>

          <div className="right-column">
            <AnswerPanel result={result} loading={loading} />
            <CitationList citations={citations} />
          </div>
        </div>
      </main>
    </div>
  );
}
