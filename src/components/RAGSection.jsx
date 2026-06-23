import React, { useState, useRef } from 'react';
import { Upload, Database, FileText, Trash2, RefreshCw, HelpCircle } from 'lucide-react';

export default function RAGSection({ 
  ragDocuments, 
  onUploadRAG, 
  onDeleteDocument, 
  isRAGLoading,
  provider,
  hasApiKey
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      onUploadRAG(files);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUploadRAG(Array.from(e.dataTransfer.files));
    }
  };

  const formatSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      {/* Upload Box */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => !isRAGLoading && fileInputRef.current?.click()}
        className={`rounded-xl p-4 text-center border transition-all duration-300 relative overflow-hidden ${
          isRAGLoading 
            ? 'border-white/5 bg-zinc-950/20 cursor-not-allowed'
            : isDragActive 
              ? 'border-indigo-500/50 bg-indigo-500/10 cursor-pointer shadow-[0_0_15px_rgba(99,102,241,0.2)]' 
              : 'border-white/5 hover:border-indigo-500/30 bg-[#09090c]/50 hover:bg-[#0d0d12]/60 cursor-pointer'
        }`}
      >
        {isRAGLoading && <div className="scanning-line bg-gradient-to-r from-transparent via-indigo-500 to-transparent" />}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf,.docx"
          multiple
          className="hidden"
          disabled={isRAGLoading}
        />

        {isRAGLoading ? (
          <div className="flex flex-col items-center justify-center space-y-2 py-3">
            <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
            <p className="text-xs font-semibold text-zinc-200 font-space uppercase tracking-wider">Vectorizing Knowledge Source...</p>
            <p className="text-xxs text-zinc-500">Parsing text, generating chunks, and computing vector embeddings</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center space-y-2 py-2 select-none">
            <Database className="w-7 h-7 text-indigo-400 filter drop-shadow-[0_0_8px_rgba(99,102,241,0.4)] animate-pulse" />
            <div>
              <p className="text-xs font-bold text-white tracking-wide">Drag & Drop knowledge documents</p>
              <p className="text-[10px] text-slate-400 mt-0.5">Supports .pdf / .docx files to chunk and index in Vector DB</p>
            </div>
          </div>
        )}
      </div>

      {/* Embedded database status message */}
      <div className="p-2.5 rounded-lg border border-white/5 bg-[#07070a]/50 text-[10px] text-zinc-400 flex items-start gap-1.5 leading-relaxed">
        <HelpCircle className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5" />
        <div>
          <strong>Vector Search Mode:</strong> {hasApiKey && (provider === 'gemini' || provider === 'openai') ? (
            <span className="text-emerald-400 font-bold">Enabled ({provider.toUpperCase()} Embeddings)</span>
          ) : (
            <span className="text-amber-400 font-bold">Local TF-IDF Search (No embedding key)</span>
          )}
          <p className="mt-0.5 text-zinc-500 text-xxs">Chunks are matched semantically against your narrative query to inject context.</p>
        </div>
      </div>

      {/* List of Uploaded Documents */}
      {ragDocuments.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <span className="text-xxs font-bold text-indigo-400 uppercase tracking-widest font-space flex items-center gap-1">
              <Database className="w-3 h-3" />
              Indexed Documents ({ragDocuments.length})
            </span>
          </div>
          
          <div className="max-h-40 overflow-y-auto space-y-1.5 scrollbar-thin pr-1">
            {ragDocuments.map((doc, idx) => (
              <div 
                key={idx}
                className="flex items-center justify-between p-2 rounded-lg bg-zinc-950/40 border border-white/5 hover:border-white/10 transition-all text-xs"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
                  <div className="min-w-0">
                    <p className="font-semibold text-zinc-200 truncate max-w-[180px]">{doc.name}</p>
                    <p className="text-[10px] text-zinc-500">
                      {formatSize(doc.size)} • <span className="text-indigo-400 font-medium">{doc.chunksCount} chunks</span>
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onDeleteDocument(doc.name)}
                  className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all cursor-pointer"
                  title="Remove from knowledge database"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
