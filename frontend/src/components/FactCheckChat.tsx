import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";

interface Message {
  id: string;
  type: "user" | "system";
  content: string;
  result?: FactCheckResult;
  timestamp: Date;
}

interface FactCheckResult {
  verdict: "Đúng" | "Sai" | "Chưa rõ";
  explanation: string;
  sources?: string[];
  confidence?: number;
}

export function FactCheckChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      type: "system",
      content:
        "Xin chào! Tôi là trợ lý kiểm tra thông tin. Hãy nhập một thông tin cần kiểm tra và tôi sẽ giúp bạn xác minh tính chính xác của nó.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input.trim(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    const claimToCheck = input.trim();
    setInput("");
    setIsLoading(true);

    // Call backend API
    try {
      const response = await fetch("http://localhost:8000/api/check", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ claim: claimToCheck }),
      });

      if (!response.ok) {
        throw new Error("Không thể kết nối đến server");
      }

      const result: FactCheckResult = await response.json();

      const systemMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "system",
        content: "Kết quả kiểm tra:",
        result,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, systemMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "system",
        content:
          error instanceof Error
            ? `Lỗi: ${error.message}`
            : "Đã xảy ra lỗi khi kiểm tra thông tin. Vui lòng thử lại.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case "Đúng":
        return "bg-green-500/10 text-green-600 border-green-500/20";
      case "Sai":
        return "bg-red-500/10 text-red-600 border-red-500/20";
      case "Chưa rõ":
        return "bg-yellow-500/10 text-yellow-600 border-yellow-500/20";
      default:
        return "bg-gray-500/10 text-gray-600 border-gray-500/20";
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case "Đúng":
        return <CheckCircle2 className="h-5 w-5" />;
      case "Sai":
        return <XCircle className="h-5 w-5" />;
      case "Chưa rõ":
        return <AlertCircle className="h-5 w-5" />;
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <div className="border-b bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50">
            Hệ thống Kiểm tra Thông tin
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Xác minh tính chính xác của thông tin với AI
          </p>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 container mx-auto px-4 py-6 overflow-hidden">
        <Card className="h-full flex flex-col shadow-xl">
          <ScrollArea className="flex-1 p-6">
            <div className="space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] ${
                      message.type === "user"
                        ? "bg-blue-500 text-white"
                        : "bg-slate-100 dark:bg-slate-800"
                    } rounded-lg p-4 space-y-3`}
                  >
                    <p className="text-sm leading-relaxed">{message.content}</p>

                    {message.result && (
                      <div className="space-y-3 pt-2 border-t border-slate-200 dark:border-slate-700">
                        {/* Verdict Badge */}
                        <div className="flex items-center gap-2">
                          <Badge
                            className={`${getVerdictColor(
                              message.result.verdict
                            )} flex items-center gap-1.5 px-3 py-1`}
                          >
                            {getVerdictIcon(message.result.verdict)}
                            <span className="font-semibold">
                              {message.result.verdict}
                            </span>
                          </Badge>
                          {message.result.confidence && (
                            <span className="text-xs text-slate-500">
                              Độ tin cậy:{" "}
                              {(message.result.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>

                        {/* Explanation */}
                        <div>
                          <h4 className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">
                            Giải thích:
                          </h4>
                          <p className="text-sm text-slate-700 dark:text-slate-300">
                            {message.result.explanation}
                          </p>
                        </div>

                        {/* Sources */}
                        {message.result.sources &&
                          message.result.sources.length > 0 && (
                            <div>
                              <h4 className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">
                                Nguồn tham khảo:
                              </h4>
                              <ul className="space-y-1">
                                {message.result.sources.map((source, idx) => (
                                  <li key={idx}>
                                    <a
                                      href={source}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-xs text-blue-600 dark:text-blue-400 hover:underline break-all"
                                    >
                                      {source}
                                    </a>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                      </div>
                    )}

                    <p className="text-xs opacity-70 mt-2">
                      {message.timestamp.toLocaleTimeString("vi-VN")}
                    </p>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        Đang kiểm tra thông tin...
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 border-t">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Nhập thông tin cần kiểm tra..."
                className="min-h-[60px] max-h-[120px] resize-none"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                disabled={isLoading}
              />
              <Button
                type="submit"
                size="icon"
                className="h-[60px] w-[60px] shrink-0"
                disabled={!input.trim() || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </Button>
            </form>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 text-center">
              Nhấn Enter để gửi, Shift + Enter để xuống dòng
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
