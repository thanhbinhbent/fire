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
    mode?: string; // "fast" or "accurate"
    latency?: string; // "5.22s"
    searches?: number; // Number of searches performed
    fast_path?: boolean; // Whether fast-path was used
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
  };
}
