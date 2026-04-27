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


export interface ConversationSummary {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_at: string;
}


export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "error";
  content: string;
  status: "streaming" | "done" | "error";
  citations: Citation[];
  model: string | null;
  retrieval_count: number | null;
  created_at: string;
}


export interface QAResponse {
  question: string;
  answer: string;
  citations: Citation[];
  model: string;
  retrieval_count: number;
}


export interface QARequest {
  user_id: string;
  conversation_id: string;
  question: string;
  top_k: number;
  history?: ConversationTurn[];
}


export type ChatMessage = StoredMessage;


export interface StreamMetaEvent {
  question: string;
  model: string;
  retrieval_count: number;
  server_started_at_ms?: number;
  retrieval_finished_at_ms?: number;
}


export interface StreamDeltaEvent {
  text: string;
  server_first_token_at_ms?: number;
}


export interface StreamDoneEvent {
  answer: string;
  server_completed_at_ms?: number;
}


export interface StreamCitationsEvent {
  citations: Citation[];
}


export interface PerformanceSample {
  request_id: string;
  question: string;
  status: "done" | "error" | "aborted";
  created_at: string;
  submit_start_at_ms: number;
  request_sent_at_ms: number | null;
  response_headers_at_ms: number | null;
  first_delta_at_ms: number | null;
  first_paint_at_ms: number | null;
  done_event_at_ms: number | null;
  final_paint_at_ms: number | null;
  server_started_at_ms: number | null;
  retrieval_finished_at_ms: number | null;
  server_first_token_at_ms: number | null;
  server_completed_at_ms: number | null;
  time_to_first_delta_ms: number | null;
  time_to_first_visible_char_ms: number | null;
  time_to_done_event_ms: number | null;
  time_to_full_visible_answer_ms: number | null;
  server_retrieval_ms: number | null;
  server_time_to_first_token_ms: number | null;
  server_total_ms: number | null;
  network_and_render_to_first_char_ms: number | null;
  network_and_render_to_full_answer_ms: number | null;
  error_message: string | null;
}


export interface AggregatedMetric {
  avg: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
}


export type PerformanceMetricKey =
  | "time_to_first_visible_char_ms"
  | "time_to_full_visible_answer_ms"
  | "server_retrieval_ms"
  | "server_time_to_first_token_ms"
  | "server_total_ms";


export interface PerformanceAggregate {
  sample_count: number;
  metrics: Partial<Record<PerformanceMetricKey, AggregatedMetric>>;
}
