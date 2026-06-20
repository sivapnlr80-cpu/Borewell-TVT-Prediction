import React, { useRef, useEffect } from 'react';
import { 
  Printer, 
  FileDown, 
  Eye, 
  FileSpreadsheet,
  Bold,
  Italic,
  Underline,
  Strikethrough,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  List,
  ListOrdered
} from 'lucide-react';
import { exportToDoc } from '../utils/docxExporter';

/**
 * Simulates a professional A4 sheet with contentEditable rich text editing
 * and a standard formatting toolbar, enabling full document editing in-browser.
 */
export default function DocumentPreview({ htmlContent, onContentChange, docType, toneMode, draftingLanguage }) {
  const editorRef = useRef(null);

  // Sync external changes (e.g. initial LLM generation or chat refinement)
  // but avoid updating innerHTML while the user is actively typing.
  useEffect(() => {
    if (editorRef.current && editorRef.current.innerHTML !== htmlContent) {
      editorRef.current.innerHTML = htmlContent;
    }
  }, [htmlContent]);

  const handleInput = (e) => {
    const newHtml = e.currentTarget.innerHTML;
    onContentChange(newHtml);
  };

  const handlePrint = () => {
    window.print();
  };

  const handleWordExport = () => {
    if (!htmlContent) return;
    
    const cleanDocType = docType.toLowerCase().replace(/[^a-z0-9]/g, '_');
    const cleanTone = toneMode.toLowerCase().replace(/[^a-z0-9]/g, '_');
    const fileName = `official_${cleanDocType}_${cleanTone}.doc`;
    
    exportToDoc(htmlContent, fileName, draftingLanguage);
  };

  // Run native document command for editing formatting
  const execCmd = (command, value = null) => {
    document.execCommand(command, false, value);
    if (editorRef.current) {
      onContentChange(editorRef.current.innerHTML);
    }
  };
  return (
    <div className="flex flex-col h-full bg-[#040406]">
      {/* Top Toolbar */}
      <div className="flex flex-col border-b border-white/5 bg-[#0a0a0e]/90 shrink-0 print:hidden">
        
        {/* Title & Action Buttons */}
        <div className="flex items-center justify-between p-3 border-b border-white/5">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-bold font-space tracking-wider text-cyan-400 uppercase">Interactive Document Editor & Preview</span>
          </div>
          
          {htmlContent && (
            <div className="flex items-center gap-2">
              <button
                onClick={handleWordExport}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium cyber-button-secondary rounded-lg transition-colors cursor-pointer"
              >
                <FileDown className="w-3.5 h-3.5" />
                Word Export
              </button>
              <button
                onClick={handlePrint}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium cyber-button-primary rounded-lg transition-all cursor-pointer"
              >
                <Printer className="w-3.5 h-3.5" />
                Print / Save PDF
              </button>
            </div>
          )}
        </div>

        {/* Rich Text Editor Toolbar */}
        {htmlContent && (
          <div className="flex flex-wrap items-center gap-1 p-2 bg-[#08080a] border-t border-white/5 px-3">
            <button
              onClick={() => execCmd('bold')}
              title="Bold"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <Bold className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('italic')}
              title="Italic"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <Italic className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('underline')}
              title="Underline"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <Underline className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('strikeThrough')}
              title="Strikethrough"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <Strikethrough className="w-3.5 h-3.5" />
            </button>

            <div className="h-4 w-px bg-zinc-800 mx-1" />

            <button
              onClick={() => execCmd('justifyLeft')}
              title="Align Left"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <AlignLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('justifyCenter')}
              title="Align Center"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <AlignCenter className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('justifyRight')}
              title="Align Right"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <AlignRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('justifyFull')}
              title="Justify"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <AlignJustify className="w-3.5 h-3.5" />
            </button>

            <div className="h-4 w-px bg-zinc-800 mx-1" />

            <button
              onClick={() => execCmd('insertUnorderedList')}
              title="Bullet List"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <List className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => execCmd('insertOrderedList')}
              title="Numbered List"
              className="p-1.5 rounded hover:bg-white/5 text-zinc-400 hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <ListOrdered className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Main Preview Container */}
      <div className="flex-1 overflow-y-auto bg-[#040406] hologram-grid p-4 flex flex-col h-full scrollbar-thin">
        {htmlContent ? (
          /* A4 Sheet Simulation container */
          <div
            id="document-preview-container"
            className="a4-page animate-fade-in text-black w-full flex flex-col"
          >
            {/* The editable canvas area */}
            <div 
              ref={editorRef}
              contentEditable={true}
              suppressContentEditableWarning={true}
              onInput={handleInput}
              className={`a4-content leading-relaxed text-justify w-full focus:outline-none w-full cursor-text ${
                draftingLanguage === 'Telugu' ? 'font-telugu' : 'font-english'
              }`}
              title="Click anywhere to edit this official document draft directly."
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center max-w-sm mx-auto self-center space-y-3 py-20">
            <div className="p-4 rounded-full bg-zinc-900 border border-white/5 text-cyan-400/80 shadow-[0_0_15px_rgba(6,182,212,0.1)]">
              <FileSpreadsheet className="w-10 h-10 animate-pulse" />
            </div>
            <div>
              <h4 className="text-sm font-bold font-space tracking-wider text-cyan-400 uppercase">No Document Drafted Yet</h4>
              <p className="text-xs text-zinc-500 mt-1">
                Upload a style reference, enter the required metadata, describe your situation, and click "Generate" to see the preview.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
