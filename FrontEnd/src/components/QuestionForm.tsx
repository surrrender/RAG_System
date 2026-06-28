import { FormEvent, KeyboardEvent, useLayoutEffect, useRef } from "react";


interface QuestionFormProps {
  question: string;
  loading: boolean;
  validationError: string | null;
  onStop: () => void;
  onQuestionChange: (value: string) => void;
  onSubmit: () => void;
}


export default function QuestionForm({
  question,
  loading,
  validationError,
  onStop,
  onQuestionChange,
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
    if (loading) {
      onStop();
      return;
    }
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
          {loading ? (
            <button type="button" className="submit-button composer-button stop-button" onClick={onStop}>
              暂停
            </button>
          ) : (
            <button type="submit" className="submit-button composer-button" disabled={!canSubmit}>
              发送问题
            </button>
          )}
        </div>
      </div>
      {validationError ? <p className="field-error composer-error">{validationError}</p> : null}
    </form>
  );
}
