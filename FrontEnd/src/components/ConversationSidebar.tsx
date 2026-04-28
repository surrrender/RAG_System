import { useState, useRef, useEffect } from "react";
import type { ConversationSummary } from "../types";

interface ConversationSidebarProps {
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  loading: boolean;
  onCreate: () => void;
  onSelect: (conversationId: string) => void;
  onRename: (conversation: ConversationSummary) => void;
  onDelete: (conversation: ConversationSummary) => void;
}

export default function ConversationSidebar({
  conversations,
  activeConversationId,
  loading,
  onCreate,
  onSelect,
  onRename,
  onDelete,
}: ConversationSidebarProps) {
  const [openDropdownId, setOpenDropdownId] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpenDropdownId(null);
      }
    }
    
    if (openDropdownId) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [openDropdownId]);

  return (
    <aside className="conversation-sidebar panel">
      <div className="conversation-sidebar-header">
        <button type="button" className="conversation-new-chat-button" onClick={onCreate} disabled={loading}>
          <span className="conversation-new-chat-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              <line x1="9" y1="10" x2="15" y2="10"></line>
              <line x1="12" y1="7" x2="12" y2="13"></line>
            </svg>
          </span>
          <span>新建会话</span>
        </button>

        <div>
          {/*<p className="eyebrow">Workspace</p>*/}
          <h2 className="conversation-sidebar-title">会话列表</h2>
        </div>
      </div>

      <div className="conversation-list">
        {conversations.map((conversation) => {
          const isActive = conversation.id === activeConversationId;
          const isDropdownOpen = openDropdownId === conversation.id;
          
          return (
            <article
              key={conversation.id}
              className={`conversation-card ${isActive ? "conversation-card-active" : ""}`}
            >
              <button
                type="button"
                className="conversation-card-button"
                aria-label={`打开会话 ${conversation.title}`}
                onClick={() => onSelect(conversation.id)}
                disabled={loading && isActive}
              >
                <div className="conversation-card-title-container">
                  <strong className="conversation-title-text">{conversation.title}</strong>
                  <span className="conversation-time-text">{formatConversationTime(conversation.last_message_at)}</span>
                </div>
              </button>
              
              <div className="conversation-card-actions" ref={isDropdownOpen ? dropdownRef : null}>
                <button
                  type="button"
                  className={`conversation-more-button ${isDropdownOpen ? "active" : ""}`}
                  aria-label="更多操作"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenDropdownId(isDropdownOpen ? null : conversation.id);
                  }}
                  disabled={loading}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5 12H5.01M12 12H12.01M19 12H19.01M6 12C6 12.5523 5.55228 13 5 13C4.44772 13 4 12.5523 4 12C4 11.4477 4.44772 11 5 11C5.55228 11 6 11.4477 6 12ZM13 12C13 12.5523 12.5523 13 12 13C11.4477 13 11 12.5523 11 12C11 11.4477 11.4477 11 12 11C12.5523 11 13 11.4477 13 12ZM20 12C20 12.5523 19.5523 13 19 13C18.4477 13 18 12.5523 18 12C18 11.4477 18.4477 11 19 11C19.5523 11 20 11.4477 20 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>

                {isDropdownOpen && (
                  <div className="conversation-dropdown-menu">
                    <button
                      type="button"
                      className="conversation-dropdown-item"
                      aria-label={`重命名会话 ${conversation.title}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onRename(conversation);
                        setOpenDropdownId(null);
                      }}
                      disabled={loading}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                      </svg>
                      重命名
                    </button>
                    <button
                      type="button"
                      className="conversation-dropdown-item text-danger"
                      aria-label={`删除会话 ${conversation.title}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(conversation);
                        setOpenDropdownId(null);
                      }}
                      disabled={loading}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                      </svg>
                      删除
                    </button>
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </aside>
  );
}


function formatConversationTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "刚刚";
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
