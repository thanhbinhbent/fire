export interface Message {
  id: string;
  type: "user" | "system";
  content: string;
  result?: FactCheckResult;
  timestamp: Date;
}

export interface FactCheckResult {
  verdict: "Đúng" | "Sai" | "Chưa rõ";
  explanation: string;
  sources?: string[];
  confidence?: number;
}
