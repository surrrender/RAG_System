import { describe, expect, it } from "vitest";

import { sanitizeMarkdownResponse } from "./markdown";


describe("sanitizeMarkdownResponse", () => {
  it("removes duplicated outer fences", () => {
    const input = "```\n```js\nconst x = 1;\n```\n```";

    expect(sanitizeMarkdownResponse(input)).toBe("```js\nconst x = 1;\n```");
  });

  it("drops stray double backtick lines", () => {
    const input = "``\nHello\n``";

    expect(sanitizeMarkdownResponse(input)).toBe("Hello");
  });
});
