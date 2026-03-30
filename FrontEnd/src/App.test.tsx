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

    expect(screen.getByRole("button", { name: "生成中..." })).toBeDisabled();
    expect(screen.getByText("流式生成中")).toBeInTheDocument();

    resolveRequest?.();

    await waitFor(() => {
      expect(screen.queryByText("流式生成中")).not.toBeInTheDocument();
    });
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
