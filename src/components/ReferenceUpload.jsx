import React, { useState, useRef } from 'react';
import { Upload, CheckCircle, RefreshCw } from 'lucide-react';
import { parseDocument } from '../utils/documentParser';

export default function ReferenceUpload({ onUploadSuccess, onError, currentReference }) {
  const [isParsing, setIsParsing] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setIsParsing(true);
    try {
      const result = await parseDocument(file);
      onUploadSuccess(result);
    } catch (err) {
      console.error(err);
      onError(err.message || 'Error parsing document.');
    } finally {
      setIsParsing(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-4">
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`rounded-2xl p-6 text-center cursor-pointer relative overflow-hidden transition-all duration-300 ${
          isDragActive 
            ? 'dropzone-neo-glass-active' 
            : 'dropzone-neo-glass'
        }`}
      >
        {isParsing && <div className="scanning-line" />}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf,.docx"
          className="hidden"
        />

        {isParsing ? (
          <div className="flex flex-col items-center justify-center space-y-2 py-2">
            <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
            <p className="text-sm font-semibold text-zinc-200">Parsing Reference...</p>
            <p className="text-xs text-zinc-500">Extracting text & structural layout</p>
          </div>
        ) : currentReference ? (
          <div className="flex flex-col items-center justify-center space-y-2 py-1">
            <CheckCircle className="w-8 h-8 text-emerald-500" />
            <p className="text-sm font-semibold text-zinc-200">Style Template Active</p>
            <p className="text-xs font-medium text-cyan-400 truncate max-w-xs">{currentReference.name}</p>
            <p className="text-xxs text-zinc-500">
              {currentReference.pages} Page(s) • Click to upload different file
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center space-y-3 py-4 select-none">
            {/* 3D Isometric Holographic Cloud Icon */}
            <svg className="w-16 h-16 filter drop-shadow-[0_0_15px_rgba(167,139,250,0.65)] animate-pulse" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="cloudGrad" x1="20" y1="20" x2="100" y2="100" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stopColor="#d8b4fe" stopOpacity="0.85"/>
                  <stop offset="50%" stopColor="#a78bfa" stopOpacity="0.75"/>
                  <stop offset="100%" stopColor="#818cf8" stopOpacity="0.85"/>
                </linearGradient>
                <linearGradient id="arrowGrad" x1="0" y1="0" x2="0" y2="50" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stopColor="#38bdf8"/>
                  <stop offset="100%" stopColor="#818cf8"/>
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
              <path d="M25 75 L60 55 L95 75 L60 95 Z" fill="rgba(167, 139, 250, 0.15)" />
              <path d="M35 50 Q45 35 60 45 Q75 35 85 50 Q95 65 75 75 Q60 80 45 75 Q25 65 35 50 Z" fill="url(#cloudGrad)" opacity="0.5" transform="translate(0, -6)" />
              <path d="M35 50 Q45 35 60 45 Q75 35 85 50 Q95 65 75 75 Q60 80 45 75 Q25 65 35 50 Z" fill="url(#cloudGrad)" filter="url(#glow)" />
              <g transform="translate(60, 58) scale(0.9) translate(-15, -22)">
                <path d="M15 2 L27 16 H20 V32 H10 V16 H3 Z" fill="#6d28d9" opacity="0.6" transform="translate(2, 3)" />
                <path d="M15 2 L27 16 H20 V32 H10 V16 H3 Z" fill="url(#arrowGrad)" />
                <path d="M15 2 L27 16 H20" stroke="#ffffff" strokeWidth="1.5" strokeLinecap="round" opacity="0.7" />
              </g>
            </svg>
            <p className="text-sm font-bold text-white tracking-wide">Drag & Drop your files here</p>
            <p className="text-xs text-slate-400">Drag & drop or browse .pdf / .docx files</p>
          </div>
        )}
      </div>
    </div>
  );
}
