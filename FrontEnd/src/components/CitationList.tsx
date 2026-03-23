import type { Citation } from "../types";


interface CitationListProps {
  citations: Citation[];
}


function truncateText(text: string): string {
  if (text.length <= 180) {
    return text;
  }
  return `${text.slice(0, 180).trim()}...`;
}


export default function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) {
    return (
      <section className="panel citation-panel">
        <div className="panel-header">
          <p className="eyebrow">引用来源</p>
          <h2>暂无引用</h2>
        </div>
        <p className="citation-empty">当前回答没有返回可展示的文档切片。</p>
      </section>
    );
  }

  return (
    <section className="panel citation-panel">
      <div className="panel-header">
        <p className="eyebrow">引用来源</p>
        <h2>检索命中的文档切片</h2>
      </div>
      <div className="citation-list">
        {citations.map((citation) => (
          <article className="citation-card" key={citation.chunk_id}>
            <div className="citation-topline">
              <h3>{citation.title || "未命名文档片段"}</h3>
              <span className="citation-score">相关度 {citation.score.toFixed(3)}</span>
            </div>
            {citation.section_path?.length ? (
              <p className="citation-path">{citation.section_path.join(" / ")}</p>
            ) : null}
            {citation.text ? <p className="citation-text">{truncateText(citation.text)}</p> : null}
            <div className="citation-footer">
              <span className="citation-id">{citation.chunk_id}</span>
              {citation.url ? (
                <a href={citation.url} target="_blank" rel="noreferrer">
                  查看原文
                </a>
              ) : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
