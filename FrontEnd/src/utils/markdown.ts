export function sanitizeMarkdownResponse(content: string): string {
  let normalized = content.replace(/\r\n?/g, "\n").trim();

  normalized = normalized.replace(/^\s*``\s*$/gm, "");

  while (
    normalized.startsWith("```") &&
    normalized.endsWith("```") &&
    normalized.slice(3, -3).trim().startsWith("```") &&
    normalized.slice(3, -3).trim().endsWith("```")
  ) {
    normalized = normalized.slice(3, -3).trim();
  }

  const fenceCount = (normalized.match(/```/g) || []).length;
  if (fenceCount % 2 === 1 && normalized.endsWith("```")) {
    normalized = normalized.slice(0, -3).trimEnd();
  }

  return normalized.trim();
}
