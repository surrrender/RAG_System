import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import App from "./App";
import { askQuestion } from "./api/client";


vi.mock("./api/client", () => ({
  askQuestion: vi.fn(),
}));


const mockedAskQuestion = vi.mocked(askQuestion);


describe("App", () => {
  beforeEach(() => {
    mockedAskQuestion.mockReset();
  });

  it("prevents submitting an empty question", async () => {
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByLabelText("问题");
    await user.clear(textarea);
    await user.click(screen.getByRole("button", { name: "开始问答" }));

    expect(mockedAskQuestion).not.toHaveBeenCalled();
    expect(screen.getByText("请输入问题后再提交。")).toBeInTheDocument();
  });

  it("renders answer metadata and citations after success", async () => {
    const user = userEvent.setup();
    mockedAskQuestion.mockResolvedValue({
      question: "App 生命周期是什么？",
      answer: "App 会触发 onLaunch、onShow 和 onHide。",
      model: "llama3.1:8b",
      retrieval_count: 2,
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

    render(<App />);
    await user.click(screen.getByRole("button", { name: "开始问答" }));

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
    let resolveRequest: ((value: Awaited<ReturnType<typeof askQuestion>>) => void) | undefined;

    mockedAskQuestion.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );

    render(<App />);
    await user.click(screen.getByRole("button", { name: "开始问答" }));

    expect(screen.getByRole("button", { name: "检索中..." })).toBeDisabled();
    expect(screen.getByText("正在检索相关文档并生成回答，请稍候。")).toBeInTheDocument();

    resolveRequest?.({
      question: "App 生命周期是什么？",
      answer: "answer",
      model: "llama3.1:8b",
      retrieval_count: 1,
      citations: [],
    });

    await waitFor(() => {
      expect(screen.getByText("answer")).toBeInTheDocument();
    });
  });

  it("shows readable error when the API fails", async () => {
    const user = userEvent.setup();
    mockedAskQuestion.mockRejectedValue(new Error("后端服务不可用"));

    render(<App />);
    await user.click(screen.getByRole("button", { name: "开始问答" }));

    await screen.findByText("后端服务不可用");
  });
});
