import { useState, useEffect, useRef } from "react";
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
    (c) => c.id === currentConversationId
  );

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentConversation?.messages]);

  const updateConversation = (id: string, updates: Partial<Conversation>) => {
    const updated = conversations.map((conv) =>
      conv.id === id ? { ...conv, ...updates, updatedAt: new Date() } : conv
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
    switch (verdict) {
      case "Đúng":
        return "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:bg-emerald-500/20 dark:text-emerald-400";
      case "Sai":
        return "bg-rose-500/10 text-rose-600 border-rose-500/20 dark:bg-rose-500/20 dark:text-rose-400";
      case "Chưa rõ":
        return "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:bg-amber-500/20 dark:text-amber-400";
      default:
        return "bg-slate-500/10 text-slate-600 border-slate-500/20";
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
                  Hệ thống Kiểm tra Thông tin
                </h1>
                <p className="text-xs md:text-sm text-muted-foreground truncate">
                  Xác minh tính chính xác của thông tin với AI
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full">
            <Card className="h-full flex flex-col shadow-sm border-none bg-background rounded-none">
              <ScrollArea className="flex-1 p-3 md:p-4 lg:p-6">
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
                          <div className="mt-4 space-y-5 animate-in fade-in slide-in-from-bottom-3 duration-500">
                            {/* Verdict Card */}
                            <div className="bg-gradient-to-br from-white to-slate-50 dark:from-slate-800 dark:to-slate-900 rounded-xl p-5 shadow-lg border border-slate-200/50 dark:border-slate-700/50">
                              <div className="flex items-start justify-between gap-4 mb-4">
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-2">
                                    <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md">
                                      {getVerdictIcon(message.result.verdict)}
                                    </div>
                                    <div>
                                      <p className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                                        Kết luận
                                      </p>
                                      <Badge
                                        className={`${getVerdictColor(
                                          message.result.verdict
                                        )} mt-1 px-3 py-1 text-sm font-bold shadow-sm`}
                                      >
                                        {message.result.verdict}
                                      </Badge>
                                    </div>
                                  </div>
                                </div>

                                {/* Confidence Score */}
                                {message.result.confidence !== undefined && (
                                  <div className="text-right">
                                    <p className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                                      Độ tin cậy
                                    </p>
                                    <div className="flex items-center gap-3">
                                      <div className="relative w-32 h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden shadow-inner">
                                        <div
                                          className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 via-blue-500 to-indigo-600 rounded-full transition-all duration-700 ease-out shadow-md"
                                          style={{
                                            width: `${
                                              message.result.confidence * 100
                                            }%`,
                                          }}
                                        />
                                      </div>
                                      <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
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

                            {/* Explanation Card */}
                            <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-md border border-slate-200/50 dark:border-slate-700/50">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="p-1.5 bg-amber-100 dark:bg-amber-900/30 rounded-lg">
                                  <svg
                                    className="w-4 h-4 text-amber-600 dark:text-amber-400"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                    />
                                  </svg>
                                </div>
                                <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300">
                                  Giải thích chi tiết
                                </h4>
                              </div>
                              <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed pl-8">
                                {message.result.explanation}
                              </p>
                            </div>

                            {/* Sources Cards */}
                            {message.result.sources &&
                              message.result.sources.length > 0 && (
                                <div className="space-y-3">
                                  <div className="flex items-center gap-2">
                                    <div className="p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                                      <svg
                                        className="w-4 h-4 text-blue-600 dark:text-blue-400"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          strokeWidth={2}
                                          d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                                        />
                                      </svg>
                                    </div>
                                    <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300">
                                      Nguồn tham khảo (
                                      {message.result.sources.length})
                                    </h4>
                                  </div>
                                  <div className="grid gap-3">
                                    {message.result.sources.map(
                                      (source, idx) => (
                                        <div
                                          key={idx}
                                          className="group bg-white dark:bg-slate-800 rounded-lg p-4 shadow-sm border border-slate-200/50 dark:border-slate-700/50 hover:shadow-md hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-200"
                                        >
                                          <div className="flex items-start gap-3">
                                            <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-sm">
                                              {idx + 1}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                              {typeof source === "string" ? (
                                                <a
                                                  href={source}
                                                  target="_blank"
                                                  rel="noopener noreferrer"
                                                  className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline break-all font-medium"
                                                >
                                                  {source}
                                                </a>
                                              ) : (
                                                <div className="space-y-2">
                                                  <a
                                                    href={source.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-sm font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:underline block group-hover:translate-x-1 transition-transform duration-200"
                                                  >
                                                    {source.title || source.url}
                                                  </a>
                                                  {source.snippet && (
                                                    <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed line-clamp-2 pl-3 border-l-2 border-slate-200 dark:border-slate-700">
                                                      {source.snippet}
                                                    </p>
                                                  )}
                                                  {(source.credibility !==
                                                    undefined ||
                                                    source.relevance !==
                                                      undefined) && (
                                                    <div className="flex gap-4 pt-2">
                                                      {source.credibility !==
                                                        undefined && (
                                                        <div className="flex items-center gap-2">
                                                          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-50 dark:bg-emerald-900/20 rounded-md">
                                                            <svg
                                                              className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400"
                                                              fill="currentColor"
                                                              viewBox="0 0 20 20"
                                                            >
                                                              <path
                                                                fillRule="evenodd"
                                                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                                                clipRule="evenodd"
                                                              />
                                                            </svg>
                                                            <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                                                              {(
                                                                source.credibility *
                                                                100
                                                              ).toFixed(0)}
                                                              % tin cậy
                                                            </span>
                                                          </div>
                                                        </div>
                                                      )}
                                                      {source.relevance !==
                                                        undefined && (
                                                        <div className="flex items-center gap-2">
                                                          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                                                            <svg
                                                              className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400"
                                                              fill="currentColor"
                                                              viewBox="0 0 20 20"
                                                            >
                                                              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                                                              <path
                                                                fillRule="evenodd"
                                                                d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm9.707 5.707a1 1 0 00-1.414-1.414L9 12.586l-1.293-1.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                                                clipRule="evenodd"
                                                              />
                                                            </svg>
                                                            <span className="text-xs font-semibold text-blue-700 dark:text-blue-300">
                                                              {(
                                                                source.relevance *
                                                                100
                                                              ).toFixed(0)}
                                                              % liên quan
                                                            </span>
                                                          </div>
                                                        </div>
                                                      )}
                                                    </div>
                                                  )}
                                                </div>
                                              )}
                                            </div>
                                          </div>
                                        </div>
                                      )
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
              </ScrollArea>

              {/* Input Area */}
              <div className="p-3 md:p-4 lg:p-6 border-t bg-slate-50 dark:bg-slate-900">
                <div className="max-w-4xl mx-auto">
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
    </div>
  );
}
