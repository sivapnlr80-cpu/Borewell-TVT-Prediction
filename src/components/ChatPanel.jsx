import React, { useState, useRef, useEffect } from 'react';
import { Send, CornerDownLeft, Sparkles, MessageSquare, Bot, User } from 'lucide-react';

export default function ChatPanel({
  chatMessages,
  onSendMessage,
  isLoading,
  hasDraft
}) {
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  // Suggestions for user to click and refine quickly
  const suggestions = [
    'Make it more stern',
    'Soften the tone',
    'Add a table for the PACS status list',
    'Extend the deadline by 2 weeks',
    'Add a list of monitoring officers'
  ];

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading && hasDraft) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleSuggestionClick = (suggestion) => {
    if (!isLoading && hasDraft) {
      onSendMessage(suggestion);
    }
  };
  return (
    <div className="flex flex-col h-full glass-panel border border-white/5 rounded-xl overflow-hidden shadow-lg glow-card">
      {/* Header */}
      <div className="flex items-center gap-2 p-3 border-b border-white/5 bg-[#0a0a0e] shrink-0">
        <MessageSquare className="w-4 h-4 text-cyan-400" />
        <span className="text-xs font-bold font-space tracking-wider text-cyan-400 uppercase">Iterative Refinement Loop</span>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4 min-h-[160px] max-h-[300px] scrollbar-thin bg-white text-zinc-800 rounded-lg m-3 shadow-inner border border-zinc-200">
        {!hasDraft ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-6 text-zinc-400">
            <Bot className="w-8 h-8 opacity-65 mb-2 text-zinc-500" />
            <p className="text-xs">Once a document is drafted, you can chat with the assistant here to modify, refine, or translate it in real-time.</p>
          </div>
        ) : chatMessages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-6 text-zinc-400">
            <Sparkles className="w-8 h-8 text-cyan-500 animate-pulse mb-2" />
            <p className="text-xs font-semibold text-zinc-800">Draft generated successfully!</p>
            <p className="text-[11px] mt-1 text-zinc-500">Ask the AI below to make it more stern, add tables, modify details, etc.</p>
          </div>
        ) : (
          chatMessages.map((msg, idx) => {
            const isUser = msg.sender === 'user';
            return (
              <div 
                key={idx} 
                className={`flex gap-2.5 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {/* Bot Icon */}
                {!isUser && (
                  <div className="w-6 h-6 rounded-full bg-cyan-50 flex items-center justify-center text-cyan-600 border border-cyan-100 shrink-0 mt-0.5">
                    <Bot className="w-3.5 h-3.5" />
                  </div>
                )}
                
                <div 
                  className={`max-w-[85%] rounded-xl p-2.5 text-xs ${
                    isUser
                      ? 'bg-cyan-50 text-cyan-950 border border-cyan-200/80 font-medium rounded-tr-none shadow-sm'
                      : 'bg-zinc-50 text-zinc-800 rounded-tl-none border border-zinc-200 shadow-sm'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  <span className={`block text-[9px] mt-1 text-right opacity-60 ${isUser ? 'text-cyan-600' : 'text-zinc-500'}`}>
                    {msg.time}
                  </span>
                </div>

                {/* User Icon */}
                {isUser && (
                  <div className="w-6 h-6 rounded-full bg-zinc-100 flex items-center justify-center text-zinc-600 border border-zinc-200 shrink-0 mt-0.5">
                    <User className="w-3.5 h-3.5" />
                  </div>
                )}
              </div>
            );
          })
        )}
        <div ref={chatEndRef} />
      </div>
      {/* Quick Suggestions */}
      {hasDraft && !isLoading && (
        <div className="px-4 py-2 border-t border-white/5 bg-[#08080a]/50 shrink-0">
          <p className="text-[10px] font-semibold text-zinc-500 mb-1.5 uppercase tracking-wider">Suggestions</p>
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map((suggestion, idx) => (
              <button
                key={idx}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-[10px] font-medium bg-[#08080a] border border-zinc-800 rounded-full px-2.5 py-1 text-zinc-400 hover:border-cyan-500/50 hover:text-cyan-400 hover:bg-cyan-500/5 transition-all cursor-pointer"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <form 
        onSubmit={handleSubmit}
        className="p-3 border-t border-white/5 flex items-center gap-2 bg-[#08080a] shrink-0"
      >
        <div className="relative flex-1">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!hasDraft || isLoading}
            placeholder={
              !hasDraft 
                ? 'Draft a document first to use chat...' 
                : isLoading 
                  ? 'Waiting for AI response...' 
                  : 'Enter refinement instructions (e.g., Change deadline to Friday)...'
            }
            className="w-full cyber-input pl-3 pr-10 py-2 text-xs focus:ring-2 focus:ring-cyan-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          />
          <span className="absolute inset-y-0 right-3 flex items-center text-zinc-500 select-none hidden md:flex">
            <CornerDownLeft className="w-3.5 h-3.5" />
          </span>
        </div>
        <button
          type="submit"
          disabled={!input.trim() || !hasDraft || isLoading}
          className={`p-2 rounded-lg transition-all shrink-0 ${
            input.trim() && hasDraft && !isLoading
              ? 'btn-red-submit cursor-pointer'
              : 'bg-zinc-900/40 border border-white/5 text-zinc-700 cursor-not-allowed'
          }`}
          aria-label="Send refinement request"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
