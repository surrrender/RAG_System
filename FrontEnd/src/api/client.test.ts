import { afterEach, describe, expect, it, vi } from "vitest";

import { streamQuestion } from "./client";


function createSseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();

  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) {
          controller.enqueue(encoder.encode(frame));
        }
        controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
      },
    },
  );
}


describe("streamQuestion", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("batches consecutive delta events before dispatching done", async () => {
    const onDelta = vi.fn();
    const onDone = vi.fn();

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      createSseResponse([
        `event: delta\ndata: ${JSON.stringify({ text: "Hello", server_first_token_at_ms: 12 })}\n\n`,
        `event: delta\ndata: ${JSON.stringify({ text: " world" })}\n\n`,
        `event: done\ndata: ${JSON.stringify({ answer: "Hello world", server_completed_at_ms: 30 })}\n\n`,
      ]),
    );

    await streamQuestion(
      {
        user_id: "user-1",
        conversation_id: "conversation-1",
        question: "test",
        top_k: 3,
      },
      {
        onDelta,
        onDone,
      },
    );

    expect(onDelta).toHaveBeenCalledTimes(1);
    expect(onDelta).toHaveBeenCalledWith({
      text: "Hello world",
      server_first_token_at_ms: 12,
    });
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDelta.mock.invocationCallOrder[0]).toBeLessThan(onDone.mock.invocationCallOrder[0]);
  });
});
