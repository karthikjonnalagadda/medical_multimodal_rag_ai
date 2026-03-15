import React, { useState, useRef, useEffect } from "react";
import { Send, Loader, AlertCircle, MessageCircle, Clock, BookOpen } from "lucide-react";
import { fetchChatResponse, fetchChatStats } from "../services/api";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ text: string; metadata?: Record<string, string> }>;
  structured?: {
    summary?: string;
    sections?: Array<{ title: string; content: string }>;
    full_text?: string;
  };
  timing?: {
    retrieval_time: number;
    llm_time: number;
    total_time: number;
  };
  timestamp: Date;
}

interface ChatStats {
  total_documents: number;
  embedding_model: string;
  vector_store_backend: string;
  status: string;
}

export function ChatbotPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<ChatStats | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch knowledge base stats on mount
  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchChatStats();
        setStats(data);
      } catch (err) {
        console.error("Failed to fetch stats:", err);
      }
    };

    loadStats();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetchChatResponse(input);

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-response`,
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        structured: response.structured,
        timing: {
          retrieval_time: response.retrieval_time,
          llm_time: response.llm_time,
          total_time: response.total_time,
        },
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to get response from chatbot";
      setError(errorMessage);

      // Add error message to chat
      const errorMsg: ChatMessage = {
        id: `msg-${Date.now()}-error`,
        role: "assistant",
        content: `⚠️ Error: ${errorMessage}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const exampleQuestions = [
    "What are the symptoms of pneumonia?",
    "How is diabetes treated?",
    "What causes chest pain?",
    "What is hypertension?",
  ];

  return (
    <div className="flex h-full gap-4 bg-background">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
                <MessageCircle className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">Medical AI Chatbot</h1>
                <p className="text-sm text-muted-foreground">
                  Ask medical questions, get AI-powered insights
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="text-muted-foreground hover:text-foreground transition-colors md:hidden"
            >
              📊
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
              <div className="text-center">
                <MessageCircle className="w-16 h-16 text-muted-foreground/30 mx-auto mb-4" />
                <h2 className="text-2xl font-semibold text-foreground mb-2">Medical AI Chatbot</h2>
                <p className="text-muted-foreground mb-6">
                  Ask medical questions and get AI-powered answers from our knowledge base.
                </p>

                {/* Disclaimer */}
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-left mb-6 max-w-md mx-auto">
                  <p className="text-sm text-amber-900">
                    <strong>⚠️ Disclaimer:</strong> This chatbot provides educational medical
                    information and is NOT a substitute for professional medical advice. Always
                    consult with qualified healthcare professionals for diagnosis, treatment, or
                    medical decisions.
                  </p>
                </div>

                {/* Example Questions */}
                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground mb-3">Try asking:</p>
                  <div className="space-y-2">
                    {exampleQuestions.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => setInput(q)}
                        className="w-full text-left px-4 py-3 rounded-lg bg-card border border-border hover:bg-accent transition-colors text-sm text-foreground"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg p-3 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-none"
                        : "bg-card border border-border text-foreground rounded-bl-none"
                    }`}
                  >
                    {/* Structured or plain text response */}
                    {msg.role === "assistant" && msg.structured?.sections && msg.structured.sections.length > 0 ? (
                      <div className="space-y-3">
                        {msg.structured.summary && (
                          <p className="text-sm leading-relaxed">{msg.structured.summary}</p>
                        )}
                        
                        {msg.structured.sections.map((section, idx) => (
                          <div key={idx} className="space-y-1">
                            <h4 className="font-semibold text-sm text-primary flex items-center gap-2">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary" />
                              {section.title}
                            </h4>
                            <div className="text-sm ml-4 space-y-1">
                              {section.content.split('\n').map((line, lineIdx) => {
                                const trimmed = line.trim();
                                if (!trimmed) return null;
                                
                                // Check if it's a bullet point
                                if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('*')) {
                                  return (
                                    <p key={lineIdx} className="text-sm leading-relaxed pl-2">
                                      • {trimmed.replace(/^[-•*]\s*/, '')}
                                    </p>
                                  );
                                }
                                
                                // Check if it's a numbered item
                                if (/^\d+\.\s/.test(trimmed)) {
                                  return (
                                    <p key={lineIdx} className="text-sm leading-relaxed pl-2">
                                      {trimmed}
                                    </p>
                                  );
                                }
                                
                                return (
                                  <p key={lineIdx} className="text-sm leading-relaxed">
                                    {trimmed}
                                  </p>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    )}

                    {/* Timing info for assistant messages */}
                    {msg.role === "assistant" && msg.timing && (
                      <div className="flex items-center gap-2 mt-3 text-xs opacity-70 border-t border-border/30 pt-2">
                        <Clock className="w-3 h-3" />
                        <span>
                          {msg.timing.total_time.toFixed(2)}s (retrieval: {msg.timing.retrieval_time.toFixed(3)}s,
                          llm: {msg.timing.llm_time.toFixed(3)}s)
                        </span>
                      </div>
                    )}

                    {/* Sources for assistant messages */}
                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-border/50">
                        <details className="text-xs">
                          <summary className="cursor-pointer font-medium flex items-center gap-1">
                            <BookOpen className="w-3 h-3" />
                            Sources ({msg.sources.length})
                          </summary>
                          <div className="mt-2 space-y-2">
                            {msg.sources.map((source, idx) => (
                              <div
                                key={idx}
                                className="text-xs bg-muted p-2 rounded border border-border/30"
                              >
                                <p className="whitespace-pre-wrap leading-relaxed">{source.text}</p>
                                {source.metadata && Object.keys(source.metadata).length > 0 && (
                                  <details className="mt-2 text-[10px]">
                                    <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                                      Metadata
                                    </summary>
                                    <div className="mt-1 space-y-0.5 pl-2 border-l border-border/30">
                                      {Object.entries(source.metadata).map(([key, value]) => (
                                        <div key={key} className="text-muted-foreground">
                                          <span className="font-medium text-foreground">{key}:</span>{" "}
                                          <span className="break-words">
                                            {typeof value === "string" ? value : JSON.stringify(value)}
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  </details>
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-card border border-border rounded-lg rounded-bl-none p-3">
                    <div className="flex items-center gap-2">
                      <Loader className="w-4 h-4 animate-spin text-primary" />
                      <span className="text-sm text-muted-foreground">Generating response...</span>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="flex justify-center">
                  <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 flex items-gap-2 max-w-md">
                    <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                    <span className="text-sm text-destructive">{error}</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-border bg-card p-4">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a medical question..."
              disabled={isLoading}
              className="flex-1 px-4 py-2 rounded-lg border border-border bg-background text-foreground placeholder-muted-foreground disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline">Send</span>
            </button>
          </form>
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="w-full mt-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Clear chat history
            </button>
          )}
        </div>
      </div>

      {/* Sidebar - Knowledge Base Info */}
      {showSidebar && (
        <div className="hidden md:flex flex-col w-72 border-l border-border bg-card p-4 space-y-4">
          <div>
            <h2 className="font-semibold text-foreground mb-3">Knowledge Base</h2>

            {stats ? (
              <div className="space-y-3">
                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-xs text-muted-foreground">Total Documents</p>
                  <p className="text-2xl font-bold text-foreground">
                    {stats.total_documents.toLocaleString()}
                  </p>
                </div>

                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-xs text-muted-foreground">Status</p>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <p className="text-sm font-medium text-foreground">Ready</p>
                  </div>
                </div>

                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-xs text-muted-foreground">Embedding Model</p>
                  <p className="text-xs font-mono text-foreground mt-1 truncate">
                    {stats.embedding_model}
                  </p>
                </div>

                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-xs text-muted-foreground">Vector Store</p>
                  <p className="text-sm font-medium text-foreground">{stats.vector_store_backend}</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Loader className="w-4 h-4 animate-spin text-primary" />
                <span className="text-sm text-muted-foreground">Loading...</span>
              </div>
            )}
          </div>

          <div className="border-t border-border pt-4">
            <h3 className="font-semibold text-foreground mb-3">Tips</h3>
            <ul className="space-y-2 text-xs text-muted-foreground">
              <li>• Be specific in your questions</li>
              <li>• Ask about symptoms, treatments, or conditions</li>
              <li>• Check the sources for more context</li>
              <li>• Response times are typically 1-2 seconds</li>
            </ul>
          </div>

          <div className="border-t border-border pt-4">
            <h3 className="font-semibold text-foreground mb-3 text-amber-600">Disclaimer</h3>
            <p className="text-xs text-muted-foreground bg-amber-50 dark:bg-amber-900/20 p-3 rounded border border-amber-200 dark:border-amber-800">
              For medical decisions, consult qualified healthcare professionals. This is educational
              information only.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
