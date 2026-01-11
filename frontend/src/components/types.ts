export interface Message {
  id: string;
  type: "user" | "system";
  content: string;
  result?: FactCheckResult;
  timestamp: Date;
}

export interface SourceInfo {
  url: string;
  title?: string;
  snippet?: string;
  credibility?: number;
  relevance?: number;
}

export interface FactCheckResult {
  verdict: string; // Can be "Đúng", "Sai", "Đúng (Chắc chắn)", etc.
  explanation: string;
  sources?: (string | SourceInfo)[];
  confidence?: number;
  metadata?: {
    preprocessing?: {
      normalized?: string;
      entities?: string[];
      token_count?: number;
    };
    verification?: {
      searches?: number;
      tokens_used?: number;
      raw_verdict?: string;
    };
    vietnamese_support?: boolean;
    mode?: string;
  };
}
