import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import App from "./App";
import {
  createConversation,
  deleteConversation,
  getConversationMessages,
  listConversations,
  renameConversation,
  streamQuestion,
} from "./api/client";


vi.mock("./api/client", () => ({
  listConversations: vi.fn(),
  createConversation: vi.fn(),
  renameConversation: vi.fn(),
  deleteConversation: vi.fn(),
  getConversationMessages: vi.fn(),
  streamQuestion: vi.fn(),
}));


const mockedListConversations = vi.mocked(listConversations);
const mockedCreateConversation = vi.mocked(createConversation);
const mockedRenameConversation = vi.mocked(renameConversation);
const mockedDeleteConversation = vi.mocked(deleteConversation);
const mockedGetConversationMessages = vi.mocked(getConversationMessages);
const mockedStreamQuestion = vi.mocked(streamQuestion);


describe("App", () => {
  beforeEach(() => {
    mockedListConversations.mockReset();
    mockedCreateConversation.mockReset();
    mockedRenameConversation.mockReset();
    mockedDeleteConversation.mockReset();
    mockedGetConversationMessages.mockReset();
    mockedStreamQuestion.mockReset();
    vi.spyOn(window, "prompt").mockReturnValue("重命名后的会话");
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(window.HTMLElement.prototype.scrollIntoView).mockClear();
    vi.mocked(window.scrollTo).mockClear();
    vi.mocked(window.requestAnimationFrame).mockClear();
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");
    window.__qaMetrics__ = undefined;

    mockedListConversations.mockResolvedValue([
      {
        id: "conversation-1",
        user_id: "user-1",
        title: "默认会话",
        created_at: "2026-04-26T00:00:00Z",
        updated_at: "2026-04-26T00:00:00Z",
        last_message_at: "2026-04-26T00:00:00Z",
      },
    ]);
    mockedGetConversationMessages.mockResolvedValue([]);
    mockedCreateConversation.mockResolvedValue({
      id: "conversation-new",
      user_id: "user-1",
      title: "新会话",
      created_at: "2026-04-26T00:00:00Z",
      updated_at: "2026-04-26T00:00:00Z",
      last_message_at: "2026-04-26T00:00:00Z",
    });
    mockedRenameConversation.mockResolvedValue({
      id: "conversation-1",
      user_id: "user-1",
      title: "重命名后的会话",
      created_at: "2026-04-26T00:00:00Z",
      updated_at: "2026-04-26T00:10:00Z",
      last_message_at: "2026-04-26T00:10:00Z",
    });

    let tick = 0;
    vi.spyOn(performance, "now").mockImplementation(() => {
      tick += 10;
      return tick;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function renderReadyApp(): Promise<void> {
    render(<App />);
    await screen.findByRole("button", { name: "打开会话 默认会话" });
    await waitFor(() => {
      expect(mockedGetConversationMessages).toHaveBeenCalledWith(expect.any(String), "conversation-1");
    });
  }

  it("creates a conversation automatically on first load when none exist", async () => {
    mockedListConversations.mockResolvedValueOnce([]);

    render(<App />);

    await waitFor(() => {
      expect(mockedCreateConversation).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByRole("button", { name: "打开会话 新会话" })).toBeInTheDocument();
  });

  it("prevents submitting an empty question", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    const textarea = screen.getByLabelText("问题");
    await user.clear(textarea);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(mockedStreamQuestion).not.toHaveBeenCalled();
    expect(screen.getByText("请输入问题后再提交。")).toBeInTheDocument();
  });

  it("renders streamed answer metadata and records the first visible latency only once", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion.mockImplementation(async (_, handlers) => {
      handlers.onResponseStarted?.();
      handlers.onMeta?.({
        question: "App 生命周期是什么？",
        model: "llama3.1:8b",
        retrieval_count: 2,
        server_started_at_ms: 0,
        retrieval_finished_at_ms: 12,
      });
      handlers.onDelta?.({ text: "App 会触发 ", server_first_token_at_ms: 28 });
      await Promise.resolve();
      handlers.onDelta?.({ text: "onLaunch、onShow 和 onHide。" });
      handlers.onCitations?.({
        citations: [
          {
            chunk_id: "chunk-1",
            score: 0.9234,
            title: "App",
            url: "https://example.com/app",
            section_path: ["框架", "生命周期"],
            text: "onLaunch 在初始化时触发，onShow 在前台显示时触发。",
          },
        ],
      });
      await Promise.resolve();
      handlers.onDone?.({ answer: "App 会触发 onLaunch、onShow 和 onHide。", server_completed_at_ms: 56 });
    });

    await renderReadyApp();
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("App 会触发 onLaunch、onShow 和 onHide。");
    expect(screen.getByText("模型：llama3.1:8b")).toBeInTheDocument();
    expect(screen.getByText("命中文档：2")).toBeInTheDocument();
    expect(screen.getByText("首字符可见")).toBeInTheDocument();
    expect(screen.getByText("完整回答可见")).toBeInTheDocument();

    await waitFor(() => {
      expect(window.__qaMetrics__?.latestSample?.status).toBe("done");
    });

    expect(mockedStreamQuestion).toHaveBeenCalledWith(
      expect.objectContaining({
        user_id: expect.any(String),
        conversation_id: "conversation-1",
        question: "小程序 App 生命周期是什么？",
      }),
      expect.any(Object),
      expect.any(Object),
    );
  });

  it("shows loading feedback while request is in flight", async () => {
    const user = userEvent.setup();
    let resolveRequest: (() => void) | undefined;

    mockedStreamQuestion.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = () => resolve();
        }),
    );

    await renderReadyApp();
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(screen.getByRole("button", { name: "暂停" })).toBeInTheDocument();
    expect(screen.getByText("流式生成中")).toBeInTheDocument();

    resolveRequest?.();

    await waitFor(() => {
      expect(screen.queryByText("流式生成中")).not.toBeInTheDocument();
    });
  });

  it("can pause an in-flight response and records an aborted sample", async () => {
    const user = userEvent.setup();
    let capturedSignal: AbortSignal | undefined;

    mockedStreamQuestion.mockImplementation(
      async (_, __, options) =>
        await new Promise((_, reject) => {
          capturedSignal = options?.signal;
          capturedSignal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );

    await renderReadyApp();
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(screen.getByRole("button", { name: "暂停" }));

    await waitFor(() => {
      expect(capturedSignal?.aborted).toBe(true);
    });
    expect(await screen.findByText("已暂停生成。")).toBeInTheDocument();
    expect(window.__qaMetrics__?.latestSample?.status).toBe("aborted");
  });

  it("shows readable error and records a failed sample when the API rejects", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion.mockRejectedValue(new Error("后端服务不可用"));

    await renderReadyApp();
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("后端服务不可用");
    expect(window.__qaMetrics__?.latestSample?.status).toBe("error");
    expect(window.__qaMetrics__?.latestSample?.error_message).toBe("后端服务不可用");
  });

  it("shows aggregate metrics in benchmark mode", async () => {
    const user = userEvent.setup();
    window.history.replaceState({}, "", "/?benchmark=1");

    mockedStreamQuestion
      .mockImplementationOnce(async (_, handlers) => {
        handlers.onResponseStarted?.();
        handlers.onMeta?.({
          question: "第一问",
          model: "llama3.1:8b",
          retrieval_count: 1,
          server_started_at_ms: 0,
          retrieval_finished_at_ms: 10,
        });
        handlers.onDelta?.({ text: "第一答", server_first_token_at_ms: 20 });
        await Promise.resolve();
        handlers.onDone?.({ answer: "第一答", server_completed_at_ms: 40 });
      })
      .mockImplementationOnce(async (_, handlers) => {
        handlers.onResponseStarted?.();
        handlers.onMeta?.({
          question: "第二问",
          model: "llama3.1:8b",
          retrieval_count: 1,
          server_started_at_ms: 0,
          retrieval_finished_at_ms: 12,
        });
        handlers.onDelta?.({ text: "第二答", server_first_token_at_ms: 24 });
        await Promise.resolve();
        handlers.onDone?.({ answer: "第二答", server_completed_at_ms: 48 });
      });

    await renderReadyApp();
    const textarea = screen.getByLabelText("问题");

    await user.clear(textarea);
    await user.type(textarea, "第一问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText("第一答");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "发送问题" })).toBeEnabled();
    });

    await user.type(screen.getByLabelText("问题"), "第二问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await screen.findByText("第二答");

    await waitFor(() => {
      expect(window.__qaMetrics__?.aggregate.sample_count).toBe(2);
    });
    expect(screen.getByText("聚合结果")).toBeInTheDocument();
  });

  it("supports switching, renaming, and deleting conversations", async () => {
    const user = userEvent.setup();
    mockedListConversations.mockResolvedValueOnce([
      {
        id: "conversation-1",
        user_id: "user-1",
        title: "默认会话",
        created_at: "2026-04-26T00:00:00Z",
        updated_at: "2026-04-26T00:00:00Z",
        last_message_at: "2026-04-26T00:00:00Z",
      },
      {
        id: "conversation-2",
        user_id: "user-1",
        title: "历史会话",
        created_at: "2026-04-25T00:00:00Z",
        updated_at: "2026-04-25T00:00:00Z",
        last_message_at: "2026-04-25T00:00:00Z",
      },
    ]);
    mockedGetConversationMessages.mockImplementation(async (_, conversationId) =>
      conversationId === "conversation-2"
        ? [
            {
              id: "message-remote-1",
              conversation_id: "conversation-2",
              role: "assistant",
              content: "这是第二个会话的历史回答。",
              status: "done",
              citations: [],
              model: "llama3.1:8b",
              retrieval_count: 1,
              created_at: "2026-04-25T00:00:00Z",
            },
          ]
        : [],
    );

    render(<App />);
    await screen.findByRole("button", { name: "打开会话 默认会话" });
    await user.click(screen.getByRole("button", { name: "打开会话 历史会话" }));

    await screen.findByText("这是第二个会话的历史回答。");

    await user.click(screen.getByRole("button", { name: "重命名会话 历史会话" }));
    expect(mockedRenameConversation).toHaveBeenCalledWith(expect.any(String), "conversation-2", "重命名后的会话");

    await user.click(screen.getByRole("button", { name: "删除会话 历史会话" }));
    expect(mockedDeleteConversation).toHaveBeenCalledWith(expect.any(String), "conversation-2");
  });
});
