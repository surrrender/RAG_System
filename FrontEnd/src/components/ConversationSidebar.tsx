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
  return (
    <aside className="conversation-sidebar panel">
      <div className="conversation-sidebar-header">
        <div>
          <p className="eyebrow">Workspace</p>
          <h2 className="conversation-sidebar-title">会话列表</h2>
        </div>
        <button type="button" className="sidebar-action-button" onClick={onCreate} disabled={loading}>
          新建
        </button>
      </div>

      <div className="conversation-list">
        {conversations.map((conversation) => {
          const isActive = conversation.id === activeConversationId;
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
                <strong>{conversation.title}</strong>
                <span>{formatConversationTime(conversation.last_message_at)}</span>
              </button>
              <div className="conversation-card-actions">
                <button
                  type="button"
                  aria-label={`重命名会话 ${conversation.title}`}
                  onClick={() => onRename(conversation)}
                  disabled={loading}
                >
                  重命名
                </button>
                <button
                  type="button"
                  aria-label={`删除会话 ${conversation.title}`}
                  onClick={() => onDelete(conversation)}
                  disabled={loading}
                >
                  删除
                </button>
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
