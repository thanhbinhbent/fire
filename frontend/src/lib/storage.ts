import type { Message } from "../components/types";

const STORAGE_KEY = "fact-check-conversations";

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export const loadConversations = (): Conversation[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored) as Conversation[];
    return parsed.map((conv) => ({
      ...conv,
      createdAt: new Date(conv.createdAt),
      updatedAt: new Date(conv.updatedAt),
      messages: conv.messages.map((msg) => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      })),
    }));
  } catch (error) {
    console.error("Error loading conversations:", error);
    return [];
  }
};

export const saveConversations = (conversations: Conversation[]) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (error) {
    console.error("Error saving conversations:", error);
  }
};

export const createNewConversation = (): Conversation => {
  const now = new Date();
  return {
    id: `conv-${Date.now()}`,
    title: "Cuộc trò chuyện mới",
    messages: [
      {
        id: "welcome",
        type: "system",
        content:
          "Xin chào! Tôi là trợ lý kiểm tra thông tin. Hãy nhập một thông tin cần kiểm tra và tôi sẽ giúp bạn xác minh tính chính xác của nó.",
        timestamp: now,
      },
    ],
    createdAt: now,
    updatedAt: now,
  };
};

export const generateConversationTitle = (firstMessage: string): string => {
  const maxLength = 50;
  if (firstMessage.length <= maxLength) return firstMessage;
  return firstMessage.substring(0, maxLength) + "...";
};
