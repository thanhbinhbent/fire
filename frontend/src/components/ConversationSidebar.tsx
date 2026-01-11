import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  MessageSquarePlus,
  MessageSquare,
  Trash2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import type { Conversation } from "@/lib/storage";
import { cn } from "@/lib/utils";

interface ConversationSidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
}

export function ConversationSidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: ConversationSidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const sortedConversations = [...conversations].sort(
    (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
  );

  if (isCollapsed) {
    return (
      <div className="w-14 md:w-16 border-r border-border bg-card flex flex-col items-center py-3 md:py-4 gap-3 md:gap-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="h-9 w-9"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onNewConversation}
          className="h-9 w-9"
        >
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="w-64 md:w-72 lg:w-80 border-r border-border bg-card flex flex-col">
      <div className="p-3 md:p-4 space-y-2 md:space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base md:text-lg font-semibold text-foreground">
            Lịch sử
          </h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="h-8 w-8"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
        <Button
          onClick={onNewConversation}
          className="w-full justify-center gap-2"
        >
          <MessageSquarePlus className="h-4 w-4" />
          <span className="hidden md:inline">Cuộc trò chuyện mới</span>
          <span className="md:hidden">Mới</span>
        </Button>
      </div>
      <Separator />
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {sortedConversations.map((conv) => (
            <div
              key={conv.id}
              className={cn(
                "group relative flex items-center gap-2 rounded-lg p-2.5 md:p-3 cursor-pointer transition-colors hover:bg-accent",
                currentConversationId === conv.id && "bg-accent"
              )}
              onClick={() => onSelectConversation(conv.id)}
            >
              <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate text-foreground">
                  {conv.title}
                </p>
                <p className="text-xs text-muted-foreground">
                  {conv.updatedAt.toLocaleDateString("vi-VN")}
                </p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(conv.id);
                }}
              >
                <Trash2 className="h-4 w-4 text-red-500" />
              </Button>
            </div>
          ))}
          {sortedConversations.length === 0 && (
            <div className="text-center py-8 text-sm text-muted-foreground">
              Chưa có cuộc trò chuyện nào
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
