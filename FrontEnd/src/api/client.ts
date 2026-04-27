import type {
  ConversationSummary,
  QARequest,
  StoredMessage,
  StreamCitationsEvent,
  StreamDeltaEvent,
  StreamDoneEvent,
  StreamMetaEvent,
} from "../types";


const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "/api";


interface StreamHandlers {
  onResponseStarted?: () => void;
  onMeta?: (payload: StreamMetaEvent) => void;
  onDelta?: (payload: StreamDeltaEvent) => void;
  onCitations?: (payload: StreamCitationsEvent) => void;
  onDone?: (payload: StreamDoneEvent) => void;
  onError?: (message: string) => void;
}

interface StreamOptions {
  signal?: AbortSignal;
}

export async function listConversations(userId: string): Promise<ConversationSummary[]> {
  const response = await fetch(`${defaultBaseUrl}/conversations?user_id=${encodeURIComponent(userId)}`);
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return (await response.json()) as ConversationSummary[];
}

export async function createConversation(userId: string): Promise<ConversationSummary> {
  const response = await fetch(`${defaultBaseUrl}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return (await response.json()) as ConversationSummary;
}

export async function renameConversation(userId: string, conversationId: string, title: string): Promise<ConversationSummary> {
  const response = await fetch(`${defaultBaseUrl}/conversations/${conversationId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ user_id: userId, title }),
  });
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return (await response.json()) as ConversationSummary;
}

export async function deleteConversation(userId: string, conversationId: string): Promise<void> {
  const response = await fetch(
    `${defaultBaseUrl}/conversations/${conversationId}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
}

export async function getConversationMessages(userId: string, conversationId: string): Promise<StoredMessage[]> {
  const response = await fetch(
    `${defaultBaseUrl}/conversations/${conversationId}/messages?user_id=${encodeURIComponent(userId)}`,
  );
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return (await response.json()) as StoredMessage[];
}

export async function streamQuestion(
  payload: QARequest,
  handlers: StreamHandlers,
  options: StreamOptions = {},
): Promise<void> {
  const response = await fetch(`${defaultBaseUrl}/qa/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream", //关键：告诉服务器我要 SSE 流式响应
    },
    signal: options.signal,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  handlers.onResponseStarted?.();

  if (!response.body) {
    throw new Error("浏览器当前环境不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n"); //把 SSE 格式的流式响应切分成一帧一帧的，SSE 规定每个事件之间是以两个换行符分隔的
    buffer = frames.pop() || "";

    // 实时处理每一条数据：更新 UI，触发回调等
    for (const frame of frames) {
      processSseFrame(frame, handlers);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    processSseFrame(buffer, handlers);
  }
}


async function getErrorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: string | { msg?: string }[] };
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail.trim();
    }
    if (Array.isArray(data.detail) && data.detail[0]?.msg) {
      return data.detail[0].msg;
    }
  } catch {
    return "请求失败，请检查后端服务是否已启动。";
  }

  return "请求失败，请稍后重试。";
}


// 把一条已经切分好的 SSE 帧解析成具体事件，然后分发给对应回调；本质上是一个路由器：后端发送什么 SSE 事件，它就把对应数据送到前端正确处理的逻辑里
function processSseFrame(frame: string, handlers: StreamHandlers): void {
  const trimmed = frame.trim();
  if (!trimmed) {
    return;
  }

  const lines = trimmed.split("\n");
  // 从帧里面找到 event：行和 data：行
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
  const dataLine = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (!event || !dataLine) {
    return;
  }

  const payload = JSON.parse(dataLine) as
    | StreamMetaEvent
    | StreamCitationsEvent
    | StreamDeltaEvent
    | StreamDoneEvent
    | { message: string };

  switch (event) {
    case "meta":
      handlers.onMeta?.(payload as StreamMetaEvent);
      break;
    case "delta":
      handlers.onDelta?.(payload as StreamDeltaEvent);
      break;
    case "citations":
      handlers.onCitations?.(payload as StreamCitationsEvent);
      break;
    case "done":
      handlers.onDone?.(payload as StreamDoneEvent);
      break;
    case "error":
      handlers.onError?.((payload as { message: string }).message);
      break;
    default:
      break;
  }
}
