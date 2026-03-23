import type { QARequest, QAResponse } from "../types";


const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "/api";


export async function askQuestion(payload: QARequest): Promise<QAResponse> {
  const response = await fetch(`${defaultBaseUrl}/qa`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  return (await response.json()) as QAResponse;
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
