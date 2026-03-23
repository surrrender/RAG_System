export interface Citation {
  chunk_id: string;
  score: number;
  title: string | null;
  url: string | null;
  section_path: string[] | null;
  text: string | null;
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
}
