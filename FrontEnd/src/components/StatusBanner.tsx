interface StatusBannerProps {
  kind: "idle" | "error";
  message: string;
}


export default function StatusBanner({ kind, message }: StatusBannerProps) {
  return <div className={`status-banner status-${kind}`}>{message}</div>;
}
