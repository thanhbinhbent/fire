import { useState, useEffect, useRef, type JSX } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Send,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Sparkles,
  User,
  Bot,
} from "lucide-react";
import type { Message, FactCheckResult } from "./types";
import { ConversationSidebar } from "./ConversationSidebar";

// Helper to render markdown-like formatting
const renderMarkdown = (text: string) => {
  if (!text) return null;

  // Split by newlines to preserve paragraphs
  const paragraphs = text.split("\n").filter((p) => p.trim());

  return paragraphs.map((paragraph, idx) => {
    // Process inline markdown
    const parts: (string | JSX.Element)[] = [];
    let lastIndex = 0;

    // Match **bold** and *italic*
    const regex = /\*\*(.+?)\*\*|\*(.+?)\*/g;
    let match;

    while ((match = regex.exec(paragraph)) !== null) {
      // Add text before match
      if (match.index > lastIndex) {
        parts.push(paragraph.slice(lastIndex, match.index));
      }

      // Add formatted text
      if (match[1]) {
        // Bold
        parts.push(
          <strong key={`${idx}-${match.index}`} className="font-semibold">
            {match[1]}
          </strong>,
        );
      } else if (match[2]) {
        // Italic
        parts.push(
          <em key={`${idx}-${match.index}`} className="italic">
            {match[2]}
          </em>,
        );
      }

      lastIndex = regex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < paragraph.length) {
      parts.push(paragraph.slice(lastIndex));
    }

    return (
      <p key={idx} className="mb-2 last:mb-0">
        {parts.length > 0 ? parts : paragraph}
      </p>
    );
  });
};

import {
  type Conversation,
  loadConversations,
  saveConversations,
  createNewConversation,
  generateConversationTitle,
} from "@/lib/storage";

export function FactCheckChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<
    string | null
  >(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [verificationMode, setVerificationMode] = useState<"fast" | "accurate">(
    "fast",
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load conversations on mount
  useEffect(() => {
    const loaded = loadConversations();
    if (loaded.length === 0) {
      const newConv = createNewConversation();
      setConversations([newConv]);
      setCurrentConversationId(newConv.id);
      saveConversations([newConv]);
    } else {
      setConversations(loaded);
      setCurrentConversationId(loaded[0].id);
    }
  }, []);

  const currentConversation = conversations.find(
    (c) => c.id === currentConversationId,
  );

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentConversation?.messages]);

  const updateConversation = (id: string, updates: Partial<Conversation>) => {
    const updated = conversations.map((conv) =>
      conv.id === id ? { ...conv, ...updates, updatedAt: new Date() } : conv,
    );
    setConversations(updated);
    saveConversations(updated);
  };

  const handleNewConversation = () => {
    const newConv = createNewConversation();
    const updated = [newConv, ...conversations];
    setConversations(updated);
    setCurrentConversationId(newConv.id);
    saveConversations(updated);
  };

  const handleDeleteConversation = (id: string) => {
    const updated = conversations.filter((c) => c.id !== id);
    setConversations(updated);
    saveConversations(updated);

    if (currentConversationId === id) {
      if (updated.length > 0) {
        setCurrentConversationId(updated[0].id);
      } else {
        handleNewConversation();
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !currentConversation) return;

    const claimToCheck = input.trim();
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: claimToCheck,
      timestamp: new Date(),
    };

    // Update messages
    const newMessages = [...currentConversation.messages, userMessage];

    // Update title if this is the first user message
    let newTitle = currentConversation.title;
    if (
      currentConversation.messages.length === 1 &&
      currentConversation.messages[0].type === "system"
    ) {
      newTitle = generateConversationTitle(claimToCheck);
    }

    updateConversation(currentConversation.id, {
      messages: newMessages,
      title: newTitle,
    });

    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/check", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          claim: claimToCheck,
          mode: verificationMode,
        }),
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

      updateConversation(currentConversation.id, {
        messages: [...newMessages, systemMessage],
      });
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

      updateConversation(currentConversation.id, {
        messages: [...newMessages, errorMessage],
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getVerdictColor = (verdict: string) => {
    const verdictLower = verdict.toLowerCase();
    if (verdictLower.includes("đúng") || verdictLower.includes("supported")) {
      return "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:bg-emerald-500/20 dark:text-emerald-400";
    } else if (
      verdictLower.includes("sai") ||
      verdictLower.includes("refuted")
    ) {
      return "bg-rose-500/10 text-rose-600 border-rose-500/20 dark:bg-rose-500/20 dark:text-rose-400";
    } else if (
      verdictLower.includes("chưa") ||
      verdictLower.includes("not enough")
    ) {
      return "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:bg-amber-500/20 dark:text-amber-400";
    } else {
      return "bg-slate-500/10 text-slate-600 border-slate-500/20";
    }
  };

  const getVerdictIcon = (verdict: string) => {
    const verdictLower = verdict.toLowerCase();
    if (verdictLower.includes("đúng") || verdictLower.includes("supported")) {
      return <CheckCircle2 className="h-5 w-5" />;
    } else if (
      verdictLower.includes("sai") ||
      verdictLower.includes("refuted")
    ) {
      return <XCircle className="h-5 w-5" />;
    } else if (
      verdictLower.includes("chưa") ||
      verdictLower.includes("not enough")
    ) {
      return <AlertCircle className="h-5 w-5" />;
    } else {
      return <AlertCircle className="h-5 w-5" />;
    }
  };

  if (!currentConversation) return null;

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <ConversationSidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-border bg-card">
          <div className="px-4 md:px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary rounded-lg">
                <Sparkles className="h-5 w-5 md:h-6 md:w-6 text-primary-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="text-lg md:text-xl font-semibold text-foreground truncate">
                  {currentConversation.title}
                </h1>
                <p className="text-xs md:text-sm text-muted-foreground truncate">
                  {currentConversation.messages.filter((m) => m.result).length}{" "}
                  kết quả đã kiểm tra
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-hidden">
          <Card className="h-full flex flex-col shadow-sm border-none bg-background rounded-none">
            <ScrollArea className="flex-1 h-0">
              <div className="p-3 md:p-4 lg:p-6">
                <div className="max-w-4xl mx-auto space-y-4">
                  {currentConversation.messages.map((message, index) => (
                    <div
                      key={message.id}
                      className={`flex gap-3 ${
                        message.type === "user"
                          ? "justify-end"
                          : "justify-start"
                      } animate-in fade-in slide-in-from-bottom-2 duration-300`}
                      style={{ animationDelay: `${index * 30}ms` }}
                    >
                      {/* Avatar for bot (left side) */}
                      {message.type === "system" && (
                        <Avatar className="h-8 w-8 shrink-0">
                          <AvatarFallback className="bg-secondary text-secondary-foreground">
                            <Bot className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                      )}

                      {/* Message Content */}
                      <div
                        className={`max-w-[85%] md:max-w-[80%] lg:max-w-[70%] ${
                          message.type === "user"
                            ? "bg-primary text-primary-foreground shadow-sm"
                            : "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-50 shadow-sm"
                        } rounded-2xl p-3 md:p-4 space-y-3`}
                      >
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">
                          {message.content}
                        </p>

                        {message.result && (
                          <div className="mt-3 space-y-3">
                            {/* Verdict & Confidence */}
                            <div className="bg-card dark:bg-card backdrop-blur-sm rounded-lg p-4 border border-border">
                              <div className="flex items-center justify-between gap-4">
                                <div className="flex items-center gap-3">
                                  <div
                                    className={`p-2 rounded-lg ${
                                      getVerdictColor(
                                        message.result.verdict,
                                      ).split(" ")[0]
                                    }`}
                                  >
                                    {getVerdictIcon(message.result.verdict)}
                                  </div>
                                  <div>
                                    <p className="text-xs text-muted-foreground">
                                      Kết luận
                                    </p>
                                    <Badge
                                      className={`${getVerdictColor(
                                        message.result.verdict,
                                      )} mt-0.5 font-semibold`}
                                    >
                                      {message.result.verdict}
                                    </Badge>
                                  </div>
                                </div>

                                {message.result.confidence !== undefined && (
                                  <div className="text-right">
                                    <p className="text-xs text-muted-foreground mb-1">
                                      Độ tin cậy
                                    </p>
                                    <div className="flex items-center gap-2">
                                      <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
                                        <div
                                          className="h-full bg-primary rounded-full transition-all duration-500"
                                          style={{
                                            width: `${
                                              message.result.confidence * 100
                                            }%`,
                                          }}
                                        />
                                      </div>
                                      <span className="text-sm font-bold text-foreground">
                                        {(
                                          message.result.confidence * 100
                                        ).toFixed(0)}
                                        %
                                      </span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>

                            {/* Explanation */}
                            <div className="bg-muted/50 backdrop-blur-sm rounded-lg p-3 border border-border">
                              <p className="text-xs font-medium text-muted-foreground mb-1.5">
                                Giải thích
                              </p>
                              <div className="text-sm text-foreground leading-relaxed">
                                {renderMarkdown(message.result.explanation)}
                              </div>
                            </div>

                            {/* Sources */}
                            {message.result.sources &&
                              message.result.sources.length > 0 && (
                                <div className="space-y-2">
                                  <p className="text-xs font-medium text-muted-foreground px-1">
                                    Nguồn tham khảo (
                                    {message.result.sources.length})
                                  </p>
                                  <div className="space-y-1.5">
                                    {message.result.sources.map(
                                      (source, idx) => (
                                        <div
                                          key={idx}
                                          className="bg-card/80 backdrop-blur-sm rounded-md p-2.5 border border-border hover:border-primary/50 transition-colors"
                                        >
                                          {typeof source === "string" ? (
                                            <a
                                              href={source}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              className="text-xs text-primary hover:underline break-all"
                                            >
                                              {source}
                                            </a>
                                          ) : (
                                            <div className="space-y-1.5">
                                              <a
                                                href={source.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-xs font-medium text-primary hover:underline block"
                                              >
                                                {source.title || source.url}
                                              </a>
                                              {source.snippet && (
                                                <p className="text-xs text-muted-foreground line-clamp-2 pl-4">
                                                  {source.snippet}
                                                </p>
                                              )}
                                              {(source.credibility !==
                                                undefined ||
                                                source.relevance !==
                                                  undefined) && (
                                                <div className="flex gap-3 text-xs pl-4">
                                                  {source.credibility !==
                                                    undefined && (
                                                    <span className="text-emerald-600 dark:text-emerald-400">
                                                      ✓{" "}
                                                      {(
                                                        source.credibility * 100
                                                      ).toFixed(0)}
                                                      % tin cậy
                                                    </span>
                                                  )}
                                                  {source.relevance !==
                                                    undefined && (
                                                    <span className="text-blue-600 dark:text-blue-400">
                                                      ⊕{" "}
                                                      {(
                                                        source.relevance * 100
                                                      ).toFixed(0)}
                                                      % liên quan
                                                    </span>
                                                  )}
                                                </div>
                                              )}
                                            </div>
                                          )}
                                        </div>
                                      ),
                                    )}
                                  </div>
                                </div>
                              )}

                            {/* Metadata Info */}
                            {message.result.metadata && (
                              <div className="bg-muted/30 backdrop-blur-sm rounded-lg p-3 border border-border">
                                <div className="flex flex-wrap gap-3 text-xs">
                                  {message.result.metadata.mode && (
                                    <div className="flex items-center gap-1.5">
                                      {message.result.metadata.mode ===
                                      "fast" ? (
                                        <Sparkles className="h-3 w-3 text-amber-500" />
                                      ) : (
                                        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                                      )}
                                      <span className="text-muted-foreground">
                                        Chế độ:{" "}
                                        <span className="font-medium text-foreground">
                                          {message.result.metadata.mode ===
                                          "fast"
                                            ? "Nhanh"
                                            : "Chính xác"}
                                        </span>
                                      </span>
                                    </div>
                                  )}
                                  {message.result.metadata.latency && (
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-muted-foreground">
                                        Thời gian:{" "}
                                        <span className="font-medium text-foreground">
                                          {message.result.metadata.latency}
                                        </span>
                                      </span>
                                    </div>
                                  )}
                                  {message.result.metadata.searches !==
                                    undefined && (
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-muted-foreground">
                                        Tìm kiếm:{" "}
                                        <span className="font-medium text-foreground">
                                          {message.result.metadata.searches}
                                        </span>
                                      </span>
                                    </div>
                                  )}
                                  {message.result.metadata.fast_path !==
                                    undefined && (
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-muted-foreground">
                                        {message.result.metadata.fast_path
                                          ? "⚡ Fast-path"
                                          : "🔄 Full search"}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        <p className="text-xs opacity-60 mt-2">
                          {message.timestamp.toLocaleTimeString("vi-VN", {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>

                      {/* Avatar for user (right side) */}
                      {message.type === "user" && (
                        <Avatar className="h-8 w-8 shrink-0">
                          <AvatarFallback className="bg-primary text-primary-foreground">
                            <User className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                      )}
                    </div>
                  ))}

                  {isLoading && (
                    <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2">
                      <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl p-3 md:p-4 shadow-sm">
                        <div className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin text-primary" />
                          <span className="text-sm text-slate-600 dark:text-slate-400">
                            Đang kiểm tra thông tin...
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="p-3 md:p-4 lg:p-6 border-t bg-slate-50 dark:bg-slate-900">
              <div className="max-w-4xl mx-auto">
                {/* Mode Selector */}
                <div className="mb-3 flex items-center gap-3 justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">
                      Chế độ:
                    </span>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant={
                          verificationMode === "fast" ? "default" : "outline"
                        }
                        onClick={() => setVerificationMode("fast")}
                        disabled={isLoading}
                        className="h-8 text-xs"
                      >
                        <Sparkles className="h-3 w-3 mr-1.5" />
                        Nhanh
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant={
                          verificationMode === "accurate"
                            ? "default"
                            : "outline"
                        }
                        onClick={() => setVerificationMode("accurate")}
                        disabled={isLoading}
                        className="h-8 text-xs"
                      >
                        <CheckCircle2 className="h-3 w-3 mr-1.5" />
                        Chính xác{" "}
                      </Button>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground hidden md:block">
                    {verificationMode === "fast"
                      ? "⚡ Nhanh với bằng chứng"
                      : "🎯 Chính xác cao"}
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="flex gap-2 md:gap-3">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Nhập thông tin cần kiểm tra..."
                    className="min-h-[56px] md:min-h-[64px] max-h-[120px] resize-none text-sm md:text-base"
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
                    className="h-[56px] w-[56px] md:h-[64px] md:w-[64px] shrink-0 bg-primary hover:bg-primary/90 text-primary-foreground"
                    disabled={!input.trim() || isLoading}
                  >
                    {isLoading ? (
                      <Loader2 className="h-5 w-5 md:h-6 md:w-6 animate-spin" />
                    ) : (
                      <Send className="h-5 w-5 md:h-6 md:w-6" />
                    )}
                  </Button>
                </form>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 text-center hidden md:block">
                  Nhấn{" "}
                  <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-xs">
                    Enter
                  </kbd>{" "}
                  để gửi •{" "}
                  <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-xs">
                    Shift + Enter
                  </kbd>{" "}
                  để xuống dòng
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
