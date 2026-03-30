import type {
  QARequest,
  StreamCitationsEvent,
  StreamMetaEvent,
} from "../types";


const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "/api";


interface StreamHandlers {
  onMeta?: (payload: StreamMetaEvent) => void;
  onDelta?: (text: string) => void;
  onCitations?: (payload: StreamCitationsEvent) => void;
  onDone?: (payload: { answer: string }) => void;
  onError?: (message: string) => void;
}


export async function streamQuestion(payload: QARequest, handlers: StreamHandlers): Promise<void> {
  const response = await fetch(`${defaultBaseUrl}/qa/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

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
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

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


function processSseFrame(frame: string, handlers: StreamHandlers): void {
  const trimmed = frame.trim();
  if (!trimmed) {
    return;
  }

  const lines = trimmed.split("\n");
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
    | { text: string }
    | { answer: string }
    | { message: string };

  switch (event) {
    case "meta":
      handlers.onMeta?.(payload as StreamMetaEvent);
      break;
    case "delta":
      handlers.onDelta?.((payload as { text: string }).text);
      break;
    case "citations":
      handlers.onCitations?.(payload as StreamCitationsEvent);
      break;
    case "done":
      handlers.onDone?.(payload as { answer: string });
      break;
    case "error":
      handlers.onError?.((payload as { message: string }).message);
      break;
    default:
      break;
  }
}
