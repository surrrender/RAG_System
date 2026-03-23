import { FormEvent } from "react";


interface QuestionFormProps {
  question: string;
  topK: number;
  loading: boolean;
  validationError: string | null;
  onQuestionChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onSubmit: () => void;
}


export default function QuestionForm({
  question,
  topK,
  loading,
  validationError,
  onQuestionChange,
  onTopKChange,
  onSubmit,
}: QuestionFormProps) {
  const canSubmit = !loading;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    onSubmit();
  };

  return (
    <section className="panel panel-form">
      <div className="panel-header">
        <p className="eyebrow">提问区</p>
        <h2>向微信小程序文档发问</h2>
      </div>
      <form className="question-form" onSubmit={handleSubmit}>
        <label className="field">
          <span className="field-label">问题</span>
          <textarea
            name="question"
            value={question}
            rows={6}
            placeholder="例如：小程序 App 生命周期是什么？"
            onChange={(event) => onQuestionChange(event.target.value)}
            disabled={loading}
          />
        </label>
        <div className="form-row">
          <label className="field field-small">
            <span className="field-label">召回数量 Top K</span>
            <input
              type="number"
              min={1}
              max={20}
              step={1}
              value={topK}
              onChange={(event) => onTopKChange(Number(event.target.value))}
              disabled={loading}
            />
          </label>
          <button type="submit" className="submit-button" disabled={!canSubmit}>
            {loading ? "检索中..." : "开始问答"}
          </button>
        </div>
        {validationError ? <p className="field-error">{validationError}</p> : null}
      </form>
    </section>
  );
}
