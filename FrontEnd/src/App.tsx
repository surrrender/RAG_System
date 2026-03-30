import { useEffect, useMemo, useRef, useState } from "react";

import { streamQuestion } from "./api/client";
import ChatMessageList from "./components/ChatMessageList";
import QuestionForm from "./components/QuestionForm";
import StatusBanner from "./components/StatusBanner";
import type { ChatMessage, ConversationTurn } from "./types";


const defaultQuestion = "小程序 App 生命周期是什么？";
const defaultTopK = 5;


export default function App() {
  const [question, setQuestion] = useState(defaultQuestion);
  const [topK, setTopK] = useState(defaultTopK);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

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
    setLoading(true);

    const userMessage = createMessage("user", normalizedQuestion, "done");
    const assistantMessage = createMessage("assistant", "", "streaming");
    const history = messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .filter((message) => message.status !== "error" && message.content.trim())
      .map<ConversationTurn>((message) => ({
        role: message.role === "user" ? "user" : "assistant",
        content: message.content,
      }));

    setMessages((current) => [...current, userMessage, assistantMessage]);

    try {
      await streamQuestion(
        {
          question: normalizedQuestion,
          top_k: topK,
          history,
        },
        {
          onMeta: ({ model, retrieval_count }) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, model, retrieval_count }
                  : message,
              ),
            );
          },
          onDelta: (text) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, content: `${message.content}${text}`, status: "streaming" }
                  : message,
              ),
            );
          },
          onCitations: ({ citations }) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id ? { ...message, citations } : message,
              ),
            );
          },
          onDone: ({ answer }) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, content: answer || message.content, status: "done" }
                  : message,
              ),
            );
          },
          onError: (message) => {
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantMessage.id
                  ? { ...item, role: "error", content: message, status: "error" }
                  : item,
              ),
            );
          },
        },
      );
      setQuestion("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "请求失败，请检查服务状态。";
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantMessage.id
            ? { ...item, role: "error", content: message, status: "error" }
            : item,
        ),
      );
    } finally {
      setLoading(false);
    }
  };

  const roundCount = useMemo(() => messages.filter((message) => message.role === "user").length, [messages]);

  return (
    <div className="app-shell">
      <div className="app-bg app-bg-top" />
      <div className="app-bg app-bg-bottom" />
      <main className="app-layout">
        <div className="chat-page">
          <div className="chat-column">
            <ChatMessageList messages={messages} loading={loading} />
            {!messages.length && !loading ? (
              <StatusBanner
                kind="idle"
                message="建议先用默认问题快速验证链路，确认后端 `POST /qa/stream` 已经启动。"
              />
            ) : null}
            {roundCount > 0 ? (
              <p className="chat-footnote">
                当前会话 {roundCount} 轮，模型回复会实时流式写入，并在消息内折叠展示引用来源。
              </p>
            ) : null}
            <div ref={messageEndRef} />
          </div>
        </div>
        <div className="composer-dock">
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
        </div>
      </main>
    </div>
  );
}


let messageCounter = 0;


function createMessage(
  role: ChatMessage["role"],
  content: string,
  status: ChatMessage["status"],
): ChatMessage {
  messageCounter += 1;
  return {
    id: `message-${messageCounter}`,
    role,
    content,
    status,
    citations: [],
    model: null,
    retrieval_count: null,
  };
}
