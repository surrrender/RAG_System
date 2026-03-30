import { FormEvent, KeyboardEvent, useLayoutEffect, useRef } from "react";


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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [question]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    onSubmit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    onSubmit();
  };

  return (
    <form className="composer-shell" onSubmit={handleSubmit}>
      <div className="composer-panel">
        <textarea
          className="composer-input"
          aria-label="问题"
          name="question"
          ref={textareaRef}
          value={question}
          rows={1}
          placeholder="给微信小程序文档提问"
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <div className="composer-footer">
          <label className="composer-topk">
            <span>Top K</span>
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
          <button type="submit" className="submit-button composer-button" disabled={!canSubmit}>
            {loading ? "生成中..." : "发送问题"}
          </button>
        </div>
      </div>
      {validationError ? <p className="field-error composer-error">{validationError}</p> : null}
    </form>
  );
}
