import http from "node:http";


const port = Number(process.env.PLAYWRIGHT_BACKEND_PORT ?? 8787);


function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


function writeEvent(response, event, data) {
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}


const server = http.createServer(async (request, response) => {
  if (request.method === "OPTIONS") {
    response.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type, Accept",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
    });
    response.end();
    return;
  }

  if (request.method !== "POST" || request.url !== "/qa/stream") {
    response.writeHead(404, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ detail: "Not found" }));
    return;
  }

  let body = "";
  request.on("data", (chunk) => {
    body += chunk;
  });

  request.on("end", async () => {
    const payload = body ? JSON.parse(body) : {};
    const question = payload.question ?? "未知问题";
    const answer = "这是用于性能基准的稳定流式回答。";

    response.writeHead(200, {
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Content-Type": "text/event-stream",
    });

    writeEvent(response, "meta", {
      question,
      model: "benchmark-mock",
      retrieval_count: 2,
      server_started_at_ms: 0,
      retrieval_finished_at_ms: 18,
    });
    await sleep(60);
    writeEvent(response, "delta", {
      text: "这是用于性能",
      server_first_token_at_ms: 64,
    });
    await sleep(70);
    writeEvent(response, "delta", {
      text: "基准的稳定流式回答。",
    });
    await sleep(50);
    writeEvent(response, "citations", {
      citations: [
        {
          chunk_id: "mock-1",
          score: 0.99,
          title: "Mock Document",
          url: "https://example.com/mock",
          section_path: ["基准", "性能"],
          text: "用于前端可见时延的稳定 mock 响应。",
        },
      ],
    });
    writeEvent(response, "done", {
      answer,
      server_completed_at_ms: 192,
    });
    response.end();
  });
});


server.listen(port, "127.0.0.1");


for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
  });
}
