export interface Citation {
  chunk_id: string;
  score: number;
  title: string | null;
  url: string | null;
  section_path: string[] | null;
  text: string | null;
}


export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}


export interface QAResponse {
  question: string;
  answer: string;
  citations: Citation[];
  model: string;
  retrieval_count: number;
}


export interface QARequest {
  question: string;
  top_k: number;
  history?: ConversationTurn[];
}


export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  status: "streaming" | "done" | "error";
  citations: Citation[];
  model: string | null;
  retrieval_count: number | null;
}


export interface StreamMetaEvent {
  question: string;
  model: string;
  retrieval_count: number;
}


export interface StreamCitationsEvent {
  citations: Citation[];
}
