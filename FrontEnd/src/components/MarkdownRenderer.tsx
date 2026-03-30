import type { ComponentPropsWithoutRef, ReactNode } from "react";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { sanitizeMarkdownResponse } from "../utils/markdown";


interface MarkdownRendererProps {
  content: string;
}


const markdownComponents: Components = {
  code(props) {
    const { className, children, ...rest } = props as ComponentPropsWithoutRef<"code"> & {
      className?: string;
      children?: ReactNode;
    };
    const language = className?.replace("language-", "") || "";
    const textContent = flattenChildren(children);
    const isBlock = Boolean(className) || textContent.includes("\n");

    if (!isBlock) {
      return (
        <code className="inline-code" {...rest}>
          {children}
        </code>
      );
    }

    return (
      <div className="code-block">
        <div className="code-block-header">
          <span>{language || "code"}</span>
        </div>
        <pre>
          <code className={className} {...rest}>
            {children}
          </code>
        </pre>
      </div>
    );
  },
  a(props) {
    return <a {...props} target="_blank" rel="noreferrer" />;
  },
};


export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {sanitizeMarkdownResponse(content)}
      </ReactMarkdown>
    </div>
  );
}


function flattenChildren(children: ReactNode): string {
  if (typeof children === "string") {
    return children;
  }
  if (typeof children === "number") {
    return String(children);
  }
  if (Array.isArray(children)) {
    return children.map(flattenChildren).join("");
  }
  return "";
}
