import type { ChatMessage } from "../types";

import MarkdownRenderer from "./MarkdownRenderer";


interface ChatMessageListProps {
  messages: ChatMessage[];
  loading: boolean;
  title: string;
  userId: string;
}


function truncateText(text: string): string {
  if (text.length <= 220) {
    return text;
  }

  return `${text.slice(0, 220).trim()}...`;
}


export default function ChatMessageList({ messages, loading, title, userId }: ChatMessageListProps) {
  if (!messages.length) {
    return (
      <section className="chat-panel chat-empty-state">
        <p className="eyebrow">RAG Chat</p>
        <h1 className="chat-title">{title}</h1>
        <p className="chat-empty-copy">
          当前用户标识 `{userId.slice(0, 8)}`，在下方输入问题后，回答会像 ChatGPT 一样从这里向上展开，并在每条消息内附带引用来源。
        </p>
        {loading ? <span className="chat-live-indicator">流式生成中</span> : null}
      </section>
    );
  }

  return (
    <section className="chat-panel">
      <div className="chat-toolbar">
        <div>
          <p className="eyebrow">RAG Chat</p>
          <h1 className="chat-title">{title}</h1>
        </div>
        {loading ? <span className="chat-live-indicator">流式生成中</span> : null}
      </div>
      <div className="message-list" aria-live="polite">
        {messages.map((message) => {
          const citationCount = message.citations.length;
          const isAssistant = message.role === "assistant";
          const isError = message.role === "error";

          return (
            <article
              className={`message-card message-${message.role} message-${message.status}`}
              key={message.id}
              data-message-id={message.id}
            >
              <div className="message-meta">
                <span className="message-role">
                  {message.role === "user"
                    ? "你"
                    : message.role === "assistant"
                      ? "文档助手"
                      : "错误提示"}
                </span>
                <span className="message-status">
                  {message.status === "streaming"
                    ? "生成中"
                    : message.status === "error"
                      ? "请求失败"
                      : "已完成"}
                </span>
              </div>

              {message.role === "user" ? (
                <p className="message-question">{message.content}</p>
              ) : (
                <MarkdownRenderer content={message.content} />
              )}

              {isAssistant && (message.model || typeof message.retrieval_count === "number") ? (
                <div className="message-tags">
                  {message.model ? <span>模型：{message.model}</span> : null}
                  {typeof message.retrieval_count === "number" ? (
                    <span>命中文档：{message.retrieval_count}</span>
                  ) : null}
                </div>
              ) : null}

              {isAssistant ? (
                <details className="citation-drawer">
                  <summary>
                    引用来源
                    <span>{citationCount > 0 ? `${citationCount} 条` : "暂无引用"}</span>
                  </summary>
                  {citationCount > 0 ? (
                    <div className="citation-list">
                      {message.citations.map((citation) => (
                        <article className="citation-card" key={citation.chunk_id}>
                          <div className="citation-topline">
                            <h3>{citation.title || "未命名文档片段"}</h3>
                            <span className="citation-score">相关度 {citation.score.toFixed(3)}</span>
                          </div>
                          {citation.section_path?.length ? (
                            <p className="citation-path">{citation.section_path.join(" / ")}</p>
                          ) : null}
                          {citation.text ? <p className="citation-text">{truncateText(citation.text)}</p> : null}
                          <div className="citation-footer">
                            <span className="citation-id">{citation.chunk_id}</span>
                            {citation.url ? (
                              <a href={citation.url} target="_blank" rel="noreferrer">
                                查看原文
                              </a>
                            ) : null}
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="citation-empty">当前回答没有返回可展示的文档切片。</p>
                  )}
                </details>
              ) : null}

              {isError ? <p className="message-error-tip">你可以调整问题表述后重新发送。</p> : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
