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
        className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer relative overflow-hidden transition-all duration-300 ${
          isDragActive 
            ? 'border-cyan-400 bg-cyan-950/10 shadow-[0_0_15px_rgba(6,182,212,0.1)]' 
            : 'border-zinc-800 hover:border-cyan-500/30 bg-transparent'
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
          <div className="flex flex-col items-center justify-center space-y-2 py-1">
            <Upload className="w-8 h-8 text-zinc-500" />
            <p className="text-sm font-semibold text-zinc-300">Upload reference style file</p>
            <p className="text-xs text-zinc-500">Drag & drop or browse .pdf / .docx files</p>
          </div>
        )}
      </div>
    </div>
  );
}
