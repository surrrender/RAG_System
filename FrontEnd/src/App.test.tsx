import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import App from "./App";
import { streamQuestion } from "./api/client";


vi.mock("./api/client", () => ({
  streamQuestion: vi.fn(),
}));


const mockedStreamQuestion = vi.mocked(streamQuestion);


describe("App", () => {
  beforeEach(() => {
    mockedStreamQuestion.mockReset();
    vi.mocked(window.HTMLElement.prototype.scrollIntoView).mockClear();
    vi.mocked(window.scrollTo).mockClear();
  });

  it("prevents submitting an empty question", async () => {
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByLabelText("问题");
    await user.clear(textarea);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(mockedStreamQuestion).not.toHaveBeenCalled();
    expect(screen.getByText("请输入问题后再提交。")).toBeInTheDocument();
  });

  it("renders streamed answer metadata and citations after success", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion.mockImplementation(async (_, handlers) => {
      handlers.onMeta?.({ question: "App 生命周期是什么？", model: "llama3.1:8b", retrieval_count: 2 });
      handlers.onDelta?.("App 会触发 ");
      handlers.onDelta?.("onLaunch、onShow 和 onHide。");
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
      handlers.onDone?.({ answer: "App 会触发 onLaunch、onShow 和 onHide。" });
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("App 会触发 onLaunch、onShow 和 onHide。");
    expect(screen.getByText("模型：llama3.1:8b")).toBeInTheDocument();
    expect(screen.getByText("命中文档：2")).toBeInTheDocument();
    expect(screen.getByText("框架 / 生命周期")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "查看原文" })).toHaveAttribute(
      "href",
      "https://example.com/app",
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

    render(<App />);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(screen.getByRole("button", { name: "暂停" })).toBeInTheDocument();
    expect(screen.getByText("流式生成中")).toBeInTheDocument();

    resolveRequest?.();

    await waitFor(() => {
      expect(screen.queryByText("流式生成中")).not.toBeInTheDocument();
    });
  });

  it("clears the composer immediately after sending", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion.mockImplementation(
      () =>
        new Promise(() => {
          return;
        }),
    );

    render(<App />);
    const textarea = screen.getByLabelText("问题") as HTMLTextAreaElement;

    await user.clear(textarea);
    await user.type(textarea, "发送后清空");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    expect(textarea.value).toBe("");
  });

  it("scrolls only once when a new question is sent", async () => {
    const user = userEvent.setup();
    const getBoundingClientRectSpy = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(function mockRect() {
        const element = this as HTMLElement;
        if (element.dataset.messageId) {
          return {
            width: 0,
            height: 120,
            top: 320,
            right: 0,
            bottom: 440,
            left: 0,
            x: 0,
            y: 320,
            toJSON: () => ({}),
          };
        }

        return {
          width: 0,
          height: 160,
          top: 0,
          right: 0,
          bottom: 160,
          left: 0,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        };
      });

    mockedStreamQuestion.mockImplementation(async (_, handlers) => {
      handlers.onDelta?.("第一段");
      handlers.onDelta?.("第二段");
      handlers.onDone?.({ answer: "第一段第二段" });
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("第一段第二段");
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    expect(window.scrollTo).toHaveBeenCalledWith({
      top: 292,
      behavior: "smooth",
    });
    getBoundingClientRectSpy.mockRestore();
  });

  it("can pause an in-flight response", async () => {
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

    render(<App />);
    await user.click(screen.getByRole("button", { name: "发送问题" }));
    await user.click(screen.getByRole("button", { name: "暂停" }));

    await waitFor(() => {
      expect(capturedSignal?.aborted).toBe(true);
    });
    expect(await screen.findByText("已暂停生成。")).toBeInTheDocument();
  });

  it("shows readable error when the API fails", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion.mockRejectedValue(new Error("后端服务不可用"));

    render(<App />);
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("后端服务不可用");
  });

  it("keeps multi-turn history in the timeline", async () => {
    const user = userEvent.setup();
    mockedStreamQuestion
      .mockImplementationOnce(async (_, handlers) => {
        handlers.onMeta?.({ question: "第一问", model: "llama3.1:8b", retrieval_count: 1 });
        handlers.onDelta?.("第一答");
        handlers.onDone?.({ answer: "第一答" });
      })
      .mockImplementationOnce(async (_, handlers) => {
        handlers.onMeta?.({ question: "第二问", model: "llama3.1:8b", retrieval_count: 1 });
        handlers.onDelta?.("第二答");
        handlers.onDone?.({ answer: "第二答" });
      });

    render(<App />);
    const textarea = screen.getByLabelText("问题");

    await user.clear(textarea);
    await user.type(textarea, "第一问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("第一答");

    await user.type(screen.getByLabelText("问题"), "第二问");
    await user.click(screen.getByRole("button", { name: "发送问题" }));

    await screen.findByText("第二答");
    expect(screen.getByText("第一问")).toBeInTheDocument();
    expect(screen.getByText("第二问")).toBeInTheDocument();
  });
});
