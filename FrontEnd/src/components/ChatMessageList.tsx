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
      <section className="chat-panel chat-empty-state" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px', textAlign: 'center' }}>
        <div style={{ maxWidth: '600px' }}>
          <div style={{ marginBottom: '24px', color: '#9ca3af' }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', display: 'block' }}>
              <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
            </svg>
          </div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#374151', marginBottom: '12px' }}>
            {title === "新对话" ? "新对话" : title}
          </h2>
          <p style={{ color: '#6b7280', fontSize: '2rem', lineHeight: 1.6 }}>
            这是建立在 <span style={{ fontWeight: 500, color: '#4b5563' }}>RAG</span> 技术之上的智能问答系统，请在下方提出您的问题。
          </p>
        </div>
        {loading ? <span className="chat-live-indicator" style={{ marginTop: '24px' }}>流式生成中</span> : null}
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
              {message.role !== "user" && (
                <div className="message-meta">
                  <span className="message-role">
                    {message.role === "assistant" ? "文档助手" : "错误提示"}
                  </span>
                  <span className="message-status">
                    {message.status === "streaming" ? (
                      <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                        <svg className="icon-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25"></circle>
                          <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        生成中
                      </span>
                    ) : message.status === "error" ? (
                      "请求失败"
                    ) : (
                      "已完成"
                    )}
                  </span>
                </div>
              )}

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
