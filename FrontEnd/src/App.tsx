import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  createConversation,
  deleteConversation,
  getConversationMessages,
  listConversations,
  renameConversation,
  streamQuestion,
} from "./api/client";
import ConversationSidebar from "./components/ConversationSidebar";
import ChatMessageList from "./components/ChatMessageList";
import QuestionForm from "./components/QuestionForm";
import StatusBanner from "./components/StatusBanner";
import type {
  ChatMessage,
  ConversationSummary,
  PerformanceSample,
  StoredMessage,
  StreamDoneEvent,
  StreamMetaEvent,
} from "./types";
import { buildPerformanceAggregate, computeDelta, computeServerDelta, roundMs } from "./utils/performance";


const defaultQuestion = "小程序 App 生命周期是什么？";
const defaultTopK = 3;
const scrollTopOffset = 28;
const localUserIdStorageKey = "rag-system:user-id";


interface ActiveRequestMetrics {
  requestId: string;
  question: string;
  createdAt: string;
  submitStartAt: number;
  requestSentAt: number | null;
  responseHeadersAt: number | null;
  firstDeltaAt: number | null;
  firstPaintAt: number | null;
  doneEventAt: number | null;
  finalPaintAt: number | null;
  serverStartedAt: number | null;
  retrievalFinishedAt: number | null;
  serverFirstTokenAt: number | null;
  serverCompletedAt: number | null;
  terminalStatus: PerformanceSample["status"] | null;
  errorMessage: string | null;
}


export default function App() {
  const [question, setQuestion] = useState(defaultQuestion);
  const [topK, setTopK] = useState(defaultTopK);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [performanceSamples, setPerformanceSamples] = useState<PerformanceSample[]>([]);
  const userIdRef = useRef(getOrCreateUserId());
  const composerDockRef = useRef<HTMLDivElement | null>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const activeRequestConversationIdRef = useRef<string | null>(null);
  const pendingScrollMessageIdRef = useRef<string | null>(null);
  const activeMetricsRef = useRef<ActiveRequestMetrics | null>(null);
  const pendingFirstPaintRequestIdRef = useRef<string | null>(null);
  const pendingFinalPaintRequestIdRef = useRef<string | null>(null);
  const firstPaintFrameRef = useRef<number | null>(null);
  const finalPaintFrameRef = useRef<number | null>(null);
  const conversationLoadIdRef = useRef(0);
  const [composerHeight, setComposerHeight] = useState(188);
  const benchmarkMode = useMemo(() => {
    return new URLSearchParams(window.location.search).get("benchmark") === "1";
  }, []);
  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId) ?? null,
    [activeConversationId, conversations],
  );

  useEffect(() => {
    void bootstrapConversations();
  }, []);

  useLayoutEffect(() => {
    const targetMessageId = pendingScrollMessageIdRef.current;
    if (!targetMessageId) {
      return;
    }

    const target = document.querySelector<HTMLElement>(`[data-message-id="${targetMessageId}"]`);
    if (!target) {
      return;
    }

    const nextTop = Math.max(0, target.getBoundingClientRect().top + window.scrollY - scrollTopOffset);
    window.scrollTo({ top: nextTop, behavior: "smooth" });
    pendingScrollMessageIdRef.current = null;
  }, [messages]);

  useLayoutEffect(() => {
    const activeMetrics = activeMetricsRef.current;
    if (!activeMetrics) {
      return;
    }

    const activeMessage = messages.find((message) => message.id === activeMetrics.requestId);
    if (!activeMessage) {
      return;
    }

    if (
      pendingFirstPaintRequestIdRef.current === activeMetrics.requestId &&
      activeMessage.content.trim() &&
      activeMetrics.firstPaintAt === null
    ) {
      if (firstPaintFrameRef.current !== null) {
        cancelAnimationFrame(firstPaintFrameRef.current);
      }
      firstPaintFrameRef.current = requestAnimationFrame(() => {
        const current = activeMetricsRef.current;
        if (!current || current.requestId !== activeMetrics.requestId || current.firstPaintAt !== null) {
          return;
        }
        current.firstPaintAt = performance.now();
        pendingFirstPaintRequestIdRef.current = null;
        firstPaintFrameRef.current = null;
      });
    }

    if (
      pendingFinalPaintRequestIdRef.current === activeMetrics.requestId &&
      (activeMessage.content.trim() || activeMessage.status !== "streaming") &&
      activeMetrics.finalPaintAt === null
    ) {
      if (finalPaintFrameRef.current !== null) {
        cancelAnimationFrame(finalPaintFrameRef.current);
      }
      finalPaintFrameRef.current = requestAnimationFrame(() => {
        const current = activeMetricsRef.current;
        if (!current || current.requestId !== activeMetrics.requestId || current.finalPaintAt !== null) {
          return;
        }
        current.finalPaintAt = performance.now();
        pendingFinalPaintRequestIdRef.current = null;
        finalPaintFrameRef.current = null;
        commitTerminalSample(current.requestId);
      });
    }
  }, [messages]);

  useLayoutEffect(() => {
    const element = composerDockRef.current;
    if (!element) {
      return;
    }

    const updateHeight = () => {
      setComposerHeight(element.getBoundingClientRect().height);
    };

    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      if (firstPaintFrameRef.current !== null) {
        cancelAnimationFrame(firstPaintFrameRef.current);
      }
      if (finalPaintFrameRef.current !== null) {
        cancelAnimationFrame(finalPaintFrameRef.current);
      }
      activeRequestRef.current?.abort();
    };
  }, []);

  const latestPerformanceSample = performanceSamples.length > 0 ? performanceSamples[performanceSamples.length - 1] : null;
  const performanceAggregate = useMemo(
    () => buildPerformanceAggregate(performanceSamples),
    [performanceSamples],
  );

  useEffect(() => {
    window.__qaMetrics__ = {
      benchmarkMode,
      latestSample: latestPerformanceSample,
      samples: performanceSamples,
      aggregate: performanceAggregate,
    };
  }, [benchmarkMode, latestPerformanceSample, performanceAggregate, performanceSamples]);

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
    if (!activeConversationId) {
      setConversationError("当前没有可用会话，请稍后重试。");
      return;
    }

    setValidationError(null);
    setConversationError(null);
    setLoading(true);
    setQuestion("");

    const userMessage = createOptimisticMessage(activeConversationId, "user", normalizedQuestion, "done");
    const assistantMessage = createOptimisticMessage(activeConversationId, "assistant", "", "streaming");
    const requestController = new AbortController();
    activeRequestRef.current = requestController;
    activeRequestConversationIdRef.current = activeConversationId;
    activeMetricsRef.current = createActiveRequestMetrics(assistantMessage.id, normalizedQuestion);

    pendingScrollMessageIdRef.current = userMessage.id;
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setConversations((current) =>
      updateConversationAfterQuestion(current, activeConversationId, normalizedQuestion),
    );

    try {
      const currentMetrics = activeMetricsRef.current;
      if (currentMetrics) {
        currentMetrics.requestSentAt = performance.now();
      }

      await streamQuestion(
        {
          user_id: userIdRef.current,
          conversation_id: activeConversationId,
          question: normalizedQuestion,
          top_k: topK,
        },
        {
          onResponseStarted: () => {
            const metrics = activeMetricsRef.current;
            if (metrics && metrics.requestId === assistantMessage.id && metrics.responseHeadersAt === null) {
              metrics.responseHeadersAt = performance.now();
            }
          },
          onMeta: (payload) => {
            syncMetaMetrics(assistantMessage.id, payload);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, model: payload.model, retrieval_count: payload.retrieval_count }
                  : message,
              ),
            );
          },
          onDelta: ({ text, server_first_token_at_ms }) => {
            const metrics = activeMetricsRef.current;
            if (metrics && metrics.requestId === assistantMessage.id && metrics.firstDeltaAt === null) {
              metrics.firstDeltaAt = performance.now();
              metrics.serverFirstTokenAt = server_first_token_at_ms ?? metrics.serverFirstTokenAt;
              pendingFirstPaintRequestIdRef.current = assistantMessage.id;
            }
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
          onDone: (payload) => {
            syncDoneMetrics(assistantMessage.id, payload);
            pendingFinalPaintRequestIdRef.current = assistantMessage.id;
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantMessage.id
                  ? { ...message, content: payload.answer || message.content, status: "done" }
                  : message,
              ),
            );
            setConversations((current) => touchConversation(current, activeConversationId));
          },
          onError: (message) => {
            markActiveRequestTerminal(assistantMessage.id, "error", message);
            pendingFinalPaintRequestIdRef.current = assistantMessage.id;
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantMessage.id
                  ? { ...item, role: "error", content: message, status: "error" }
                  : item,
              ),
            );
          },
        },
        { signal: requestController.signal },
      );
    } catch (error) {
      if (isAbortError(error)) {
        markActiveRequestTerminal(assistantMessage.id, "aborted", "已暂停生成。");
        pendingFinalPaintRequestIdRef.current = assistantMessage.id;
        setMessages((current) =>
          current.map((item) =>
            item.id === assistantMessage.id
              ? {
                  ...item,
                  content: item.content || "已暂停生成。",
                  status: "done",
                }
              : item,
          ),
        );
      } else {
        const message = error instanceof Error ? error.message : "请求失败，请检查服务状态。";
        markActiveRequestTerminal(assistantMessage.id, "error", message);
        pendingFinalPaintRequestIdRef.current = assistantMessage.id;
        setMessages((current) =>
          current.map((item) =>
            item.id === assistantMessage.id
              ? { ...item, role: "error", content: message, status: "error" }
              : item,
          ),
        );
      }
    } finally {
      activeRequestRef.current = null;
      activeRequestConversationIdRef.current = null;
      setLoading(false);
    }
  };

  const handleStop = () => {
    activeRequestRef.current?.abort();
  };

  const handleSelectConversation = async (conversationId: string) => {
    if (conversationId === activeConversationId) {
      return;
    }
    await openConversation(conversationId);
  };

  const handleCreateConversation = async () => {
    setConversationError(null);
    const conversation = await createConversation(userIdRef.current);
    setConversations((current) => [conversation, ...current]);
    await openConversation(conversation.id, [conversation, ...conversations]);
  };

  const handleRenameConversation = async (conversation: ConversationSummary) => {
    const nextTitle = window.prompt("输入新的会话标题", conversation.title);
    if (!nextTitle) {
      return;
    }
    const renamed = await renameConversation(userIdRef.current, conversation.id, nextTitle);
    setConversations((current) => current.map((item) => (item.id === renamed.id ? renamed : item)));
  };

  const handleDeleteConversation = async (conversation: ConversationSummary) => {
    const confirmed = window.confirm(`确定删除会话“${conversation.title}”吗？`);
    if (!confirmed) {
      return;
    }

    if (conversation.id === activeRequestConversationIdRef.current) {
      activeRequestRef.current?.abort();
    }

    await deleteConversation(userIdRef.current, conversation.id);
    const remaining = conversations.filter((item) => item.id !== conversation.id);
    if (remaining.length === 0) {
      const created = await createConversation(userIdRef.current);
      setConversations([created]);
      await openConversation(created.id, [created]);
      return;
    }

    setConversations(remaining);
    if (conversation.id === activeConversationId) {
      await openConversation(remaining[0].id, remaining);
    }
  };

  const roundCount = useMemo(() => messages.filter((message) => message.role === "user").length, [messages]);

  return (
    <div className="app-shell">
      <main className="app-layout">
        <div className="workspace-grid">
          <ConversationSidebar
            conversations={conversations}
            activeConversationId={activeConversationId}
            loading={loading}
            onCreate={() => {
              void handleCreateConversation().catch(handleConversationFailure);
            }}
            onSelect={(conversationId) => {
              void handleSelectConversation(conversationId).catch(handleConversationFailure);
            }}
            onRename={(conversation) => {
              void handleRenameConversation(conversation).catch(handleConversationFailure);
            }}
            onDelete={(conversation) => {
              void handleDeleteConversation(conversation).catch(handleConversationFailure);
            }}
          />

          <div className="chat-page">
            <div className="chat-column">
              {conversationError ? <StatusBanner kind="error" message={conversationError} /> : null}
              <ChatMessageList
                messages={messages}
                loading={loading}
                title={activeConversation?.title ?? "新会话"}
                userId={userIdRef.current}
              />
              {/* {roundCount > 0 ? (
                <p className="chat-footnote">
                  当前会话 {roundCount} 轮，模型回复会实时流式写入，并在消息内折叠展示引用来源。
                </p>
              ) : null} */}
            </div>
            <div className="composer-dock" ref={composerDockRef}>
              <QuestionForm
                question={question}
                topK={topK}
                loading={loading || bootstrapping || !activeConversationId}
                validationError={validationError}
                onStop={handleStop}
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
          </div>
        </div>
      </main>
    </div>
  );

  async function bootstrapConversations(): Promise<void> {
    try {
      setBootstrapping(true);
      setConversationError(null);
      const items = await listConversations(userIdRef.current);
      if (items.length === 0) {
        const created = await createConversation(userIdRef.current);
        setConversations([created]);
        await openConversation(created.id, [created]);
        return;
      }
      setConversations(items);
      await openConversation(items[0].id, items);
    } catch (error) {
      handleConversationFailure(error);
    } finally {
      setBootstrapping(false);
    }
  }

  async function openConversation(
    conversationId: string,
    nextConversations: ConversationSummary[] = conversations,
  ): Promise<void> {
    activeRequestRef.current?.abort();
    setActiveConversationId(conversationId);
    setMessages([]);
    setConversationError(null);

    const loadId = conversationLoadIdRef.current + 1;
    conversationLoadIdRef.current = loadId;
    const loadedMessages = await getConversationMessages(userIdRef.current, conversationId);
    if (conversationLoadIdRef.current !== loadId) {
      return;
    }
    setConversations(nextConversations);
    setMessages(normalizeLoadedMessages(loadedMessages));
  }

  function syncMetaMetrics(requestId: string, payload: StreamMetaEvent): void {
    const metrics = activeMetricsRef.current;
    if (!metrics || metrics.requestId !== requestId) {
      return;
    }
    metrics.serverStartedAt = payload.server_started_at_ms ?? metrics.serverStartedAt;
    metrics.retrievalFinishedAt = payload.retrieval_finished_at_ms ?? metrics.retrievalFinishedAt;
  }

  function syncDoneMetrics(requestId: string, payload: StreamDoneEvent): void {
    const metrics = activeMetricsRef.current;
    if (!metrics || metrics.requestId !== requestId) {
      return;
    }
    metrics.doneEventAt = performance.now();
    metrics.serverCompletedAt = payload.server_completed_at_ms ?? metrics.serverCompletedAt;
    metrics.terminalStatus = "done";
  }

  function markActiveRequestTerminal(
    requestId: string,
    status: PerformanceSample["status"],
    errorMessage: string | null,
  ): void {
    const metrics = activeMetricsRef.current;
    if (!metrics || metrics.requestId !== requestId) {
      return;
    }
    metrics.terminalStatus = status;
    metrics.errorMessage = errorMessage;
  }

  function commitTerminalSample(requestId: string): void {
    const metrics = activeMetricsRef.current;
    if (!metrics || metrics.requestId !== requestId || !metrics.terminalStatus || metrics.finalPaintAt === null) {
      return;
    }

    const sample = buildPerformanceSample(metrics);
    setPerformanceSamples((current) => [...current, sample]);
    activeMetricsRef.current = null;
  }

  function handleConversationFailure(error: unknown): void {
    const message = error instanceof Error ? error.message : "会话操作失败，请稍后重试。";
    setConversationError(message);
    setLoading(false);
  }
}


function createActiveRequestMetrics(requestId: string, normalizedQuestion: string): ActiveRequestMetrics {
  return {
    requestId,
    question: normalizedQuestion,
    createdAt: new Date().toISOString(),
    submitStartAt: performance.now(),
    requestSentAt: null,
    responseHeadersAt: null,
    firstDeltaAt: null,
    firstPaintAt: null,
    doneEventAt: null,
    finalPaintAt: null,
    serverStartedAt: null,
    retrievalFinishedAt: null,
    serverFirstTokenAt: null,
    serverCompletedAt: null,
    terminalStatus: null,
    errorMessage: null,
  };
}


function buildPerformanceSample(metrics: ActiveRequestMetrics): PerformanceSample {
  const serverRetrievalMs = computeServerDelta(metrics.serverStartedAt, metrics.retrievalFinishedAt);
  const serverTimeToFirstTokenMs = computeServerDelta(metrics.serverStartedAt, metrics.serverFirstTokenAt);
  const serverTotalMs = computeServerDelta(metrics.serverStartedAt, metrics.serverCompletedAt);
  const timeToFirstVisibleCharMs = computeDelta(metrics.submitStartAt, metrics.firstPaintAt);
  const timeToFullVisibleAnswerMs = computeDelta(metrics.submitStartAt, metrics.finalPaintAt);

  return {
    request_id: metrics.requestId,
    question: metrics.question,
    status: metrics.terminalStatus ?? "error",
    created_at: metrics.createdAt,
    submit_start_at_ms: roundMs(metrics.submitStartAt),
    request_sent_at_ms: metrics.requestSentAt !== null ? roundMs(metrics.requestSentAt) : null,
    response_headers_at_ms: metrics.responseHeadersAt !== null ? roundMs(metrics.responseHeadersAt) : null,
    first_delta_at_ms: metrics.firstDeltaAt !== null ? roundMs(metrics.firstDeltaAt) : null,
    first_paint_at_ms: metrics.firstPaintAt !== null ? roundMs(metrics.firstPaintAt) : null,
    done_event_at_ms: metrics.doneEventAt !== null ? roundMs(metrics.doneEventAt) : null,
    final_paint_at_ms: metrics.finalPaintAt !== null ? roundMs(metrics.finalPaintAt) : null,
    server_started_at_ms: metrics.serverStartedAt,
    retrieval_finished_at_ms: metrics.retrievalFinishedAt,
    server_first_token_at_ms: metrics.serverFirstTokenAt,
    server_completed_at_ms: metrics.serverCompletedAt,
    time_to_first_delta_ms: computeDelta(metrics.submitStartAt, metrics.firstDeltaAt),
    time_to_first_visible_char_ms: timeToFirstVisibleCharMs,
    time_to_done_event_ms: computeDelta(metrics.submitStartAt, metrics.doneEventAt),
    time_to_full_visible_answer_ms: timeToFullVisibleAnswerMs,
    server_retrieval_ms: serverRetrievalMs,
    server_time_to_first_token_ms: serverTimeToFirstTokenMs,
    server_total_ms: serverTotalMs,
    network_and_render_to_first_char_ms:
      timeToFirstVisibleCharMs !== null && serverTimeToFirstTokenMs !== null
        ? roundMs(timeToFirstVisibleCharMs - serverTimeToFirstTokenMs)
        : null,
    network_and_render_to_full_answer_ms:
      timeToFullVisibleAnswerMs !== null && serverTotalMs !== null
        ? roundMs(timeToFullVisibleAnswerMs - serverTotalMs)
        : null,
    error_message: metrics.errorMessage,
  };
}


function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}


let messageCounter = 0;


function createOptimisticMessage(
  conversationId: string,
  role: ChatMessage["role"],
  content: string,
  status: ChatMessage["status"],
): ChatMessage {
  messageCounter += 1;
  return {
    id: `message-${messageCounter}`,
    conversation_id: conversationId,
    role,
    content,
    status,
    citations: [],
    model: null,
    retrieval_count: null,
    created_at: new Date().toISOString(),
  };
}


function getOrCreateUserId(): string {
  const existing = window.localStorage.getItem(localUserIdStorageKey)?.trim();
  if (existing) {
    return existing;
  }

  const nextValue = window.crypto?.randomUUID?.() ?? `user-${Date.now()}`;
  window.localStorage.setItem(localUserIdStorageKey, nextValue);
  return nextValue;
}


function normalizeLoadedMessages(messages: StoredMessage[]): ChatMessage[] {
  return messages.map((message) => ({
    ...message,
    role: message.status === "error" ? "error" : message.role,
  }));
}


function updateConversationAfterQuestion(
  conversations: ConversationSummary[],
  activeConversationId: string,
  question: string,
): ConversationSummary[] {
  return touchConversation(
    conversations.map((item) =>
      item.id === activeConversationId && item.title === "新会话"
        ? { ...item, title: question.trim().slice(0, 30) || "新会话" }
        : item,
    ),
    activeConversationId,
  );
}


function touchConversation(conversations: ConversationSummary[], conversationId: string): ConversationSummary[] {
  const timestamp = new Date().toISOString();
  const nextItems = conversations.map((item) =>
    item.id === conversationId
      ? { ...item, updated_at: timestamp, last_message_at: timestamp }
      : item,
  );
  nextItems.sort((left, right) => right.last_message_at.localeCompare(left.last_message_at));
  return nextItems;
}
