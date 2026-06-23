import React, { useState, useEffect } from 'react';
import { Settings, Moon, Sun, FileText, AlertTriangle, HelpCircle } from 'lucide-react';
import ReferenceUpload from './components/ReferenceUpload';
import DocumentForm from './components/DocumentForm';
import DocumentPreview from './components/DocumentPreview';
import ChatPanel from './components/ChatPanel';
import SettingsModal from './components/SettingsModal';
import Toast from './components/Toast';
import { callLLM } from './utils/llm';
import RAGSection from './components/RAGSection';
import { chunkText, getEmbeddings, retrieveRelevantContext } from './utils/ragEngine';
import { parseDocument } from './utils/documentParser';

export default function App() {
  // Theme state (Dark Mode / Light Mode)
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  // Settings state (LLM Provider, API Key, Model)
  const [settings, setSettings] = useState(() => {
    // Attempt to load settings
    const saved = localStorage.getItem('app_llm_settings');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Load key from localStorage specific to provider if it was saved
        const providerKey = localStorage.getItem(`api_key_${parsed.provider}`) || parsed.apiKey || '';
        return { ...parsed, apiKey: providerKey };
      } catch (e) {
        console.error('Failed to parse settings', e);
      }
    }

    // Default settings
    const defaultProvider = 'gemini';
    const defaultKey = localStorage.getItem(`api_key_${defaultProvider}`) || '';
    return {
      provider: defaultProvider,
      apiKey: defaultKey,
      model: 'gemini-2.5-flash'
    };
  });

  // Modal & Toast States
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [toast, setToast] = useState(null);

  // Document Generator States
  const [currentReference, setCurrentReference] = useState(null);
  const [docType, setDocType] = useState('Memo'); // Default to Memo matching example PDF
  const [formData, setFormData] = useState({
    from: '',
    to: '',
    officerName: 'SRI K. VENKATESWARLU, M.A.',
    designation: 'Divisional Cooperative Officer, Nellore',
    rcNo: '',
    docDate: ''
  });
  const [draftingLanguage, setDraftingLanguage] = useState('English');
  const [toneMode, setToneMode] = useState('Strict Official');
  const [narrative, setNarrative] = useState('');
  
  // Document preview content
  const [generatedDocument, setGeneratedDocument] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // RAG States
  const [ragDocuments, setRagDocuments] = useState([]);
  const [ragChunks, setRagChunks] = useState([]);
  const [isRAGLoading, setIsRAGLoading] = useState(false);

  // Sync theme to root HTML element
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  // Toast Helper
  const showToast = (message, type = 'info') => {
    setToast({ message, type, id: Date.now() });
  };

  const handleUploadSuccess = (result) => {
    setCurrentReference(result);
    showToast(`Style template successfully parsed from "${result.name}"!`, 'success');
  };

  const handleParserError = (errorMessage) => {
    showToast(errorMessage, 'error');
  };

  // Handle RAG Document upload, chunking, and embedding
  const handleUploadRAG = async (files) => {
    setIsRAGLoading(true);
    let newDocs = [...ragDocuments];
    let newChunks = [...ragChunks];
    let processedCount = 0;
    let failedCount = 0;

    try {
      for (const file of files) {
        if (newDocs.some(d => d.name === file.name)) {
          showToast(`File "${file.name}" is already in the knowledge base.`, 'warning');
          continue;
        }

        try {
          const parsed = await parseDocument(file);
          const chunks = chunkText(parsed.text);
          
          if (chunks.length === 0) {
            showToast(`Document "${file.name}" has no extractable text.`, 'warning');
            continue;
          }

          let embeddings = null;
          if (settings.apiKey && (settings.provider === 'gemini' || settings.provider === 'openai')) {
            showToast(`Vectorizing "${file.name}" using LLM Embeddings...`, 'info');
            embeddings = await getEmbeddings(chunks, settings.provider, settings.apiKey);
          }

          const chunkObjs = chunks.map((text, idx) => ({
            id: `${file.name}-${idx}`,
            docName: file.name,
            text: text,
            embedding: embeddings ? embeddings[idx] : null
          }));

          newChunks = newChunks.concat(chunkObjs);
          newDocs.push({
            name: file.name,
            size: file.size,
            chunksCount: chunks.length
          });
          processedCount++;
        } catch (e) {
          console.error("Failed to parse file:", file.name, e);
          failedCount++;
        }
      }

      setRagDocuments(newDocs);
      setRagChunks(newChunks);

      if (processedCount > 0) {
        showToast(`Successfully chunked and indexed ${processedCount} document(s) in Vector DB!`, 'success');
      }
      if (failedCount > 0) {
        showToast(`Failed to parse ${failedCount} document(s).`, 'error');
      }
    } catch (err) {
      console.error(err);
      showToast('Error vectorizing knowledge sources.', 'error');
    } finally {
      setIsRAGLoading(false);
    }
  };

  const handleDeleteRAGDocument = (docName) => {
    setRagDocuments(prev => prev.filter(d => d.name !== docName));
    setRagChunks(prev => prev.filter(c => c.docName !== docName));
    showToast(`Removed "${docName}" from knowledge database.`, 'info');
  };

  // Generate Initial Document Draft
  const handleGenerateDocument = async () => {
    if (!currentReference) {
      showToast('Please upload or select a reference template first.', 'warning');
      return;
    }
    if (!settings.apiKey) {
      setIsSettingsOpen(true);
      showToast('Please configure your API Key in the settings panel to begin.', 'warning');
      return;
    }
    if (!narrative.trim()) {
      showToast('Please enter a situation narrative.', 'warning');
      return;
    }

    setIsLoading(true);
    setChatMessages([]); // Reset chat history on a new generation
    
    try {
      let retrievedContext = [];
      if (ragChunks.length > 0) {
        showToast('Retrieving context from Vector DB...', 'info');
        retrievedContext = await retrieveRelevantContext(
          narrative,
          ragChunks,
          settings.provider,
          settings.apiKey,
          5
        );
      }

      const docHtml = await callLLM({
        provider: settings.provider,
        apiKey: settings.apiKey,
        model: settings.model,
        referenceText: currentReference.text,
        docType,
        metadata: formData,
        draftingLanguage,
        toneMode,
        narrative,
        ragContext: retrievedContext.map(c => c.text)
      });

      setGeneratedDocument(docHtml);
      showToast('Document draft generated successfully!', 'success');
    } catch (err) {
      console.error(err);
      showToast(err.message || 'Draft generation failed.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Chat Refinement Loop
  const handleSendChatMessage = async (messageText) => {
    if (!generatedDocument) return;
    if (!settings.apiKey) {
      setIsSettingsOpen(true);
      showToast('API Key is required to call AI.', 'warning');
      return;
    }

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg = { sender: 'user', text: messageText, time: timestamp };
    
    setChatMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      let retrievedContext = [];
      if (ragChunks.length > 0) {
        const query = `${narrative} \nRefinement: ${messageText}`;
        retrievedContext = await retrieveRelevantContext(
          query,
          ragChunks,
          settings.provider,
          settings.apiKey,
          5
        );
      }

      const updatedDocHtml = await callLLM({
        provider: settings.provider,
        apiKey: settings.apiKey,
        model: settings.model,
        referenceText: currentReference.text,
        docType,
        metadata: formData,
        draftingLanguage,
        toneMode,
        narrative,
        chatHistory: [...chatMessages, userMsg],
        refinementRequest: messageText,
        ragContext: retrievedContext.map(c => c.text)
      });

      setGeneratedDocument(updatedDocHtml);
      
      const botMsg = { 
        sender: 'assistant', 
        text: 'I have updated the document draft in the preview panel based on your request.', 
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
      };
      setChatMessages(prev => [...prev, botMsg]);
      showToast('Document updated!', 'success');
    } catch (err) {
      console.error(err);
      showToast(err.message || 'Refinement failed.', 'error');
      
      const botMsg = { 
        sender: 'assistant', 
        text: `Error updating document: ${err.message}`, 
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
      };
      setChatMessages(prev => [...prev, botMsg]);
    } finally {
      setIsLoading(false);
    }
  };
  const handleSaveSettings = (newSettings) => {
    setSettings(newSettings);
    localStorage.setItem('app_llm_settings', JSON.stringify(newSettings));
    showToast('AI Settings updated successfully!', 'success');
  };

  return (
    <div className="min-h-screen bg-transparent flex flex-col transition-colors duration-300 font-plus text-zinc-400">
      
      {/* Header Row */}
      <header className="bg-slate-950/20 backdrop-blur-md border-b border-white/10 px-6 py-4 shrink-0 shadow-lg print:hidden sticky top-0 z-40">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <img src="/logo.png" alt="Official Drafter Logo" className="w-24 h-24 object-contain rounded-2xl border-2 border-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.3)] bg-[#07070a] p-1.5" />
            <div>
              <h1 className="text-2xl font-extrabold font-space tracking-tight text-white bg-gradient-to-r from-[#ff9933] via-white to-[#138808] bg-clip-text text-transparent drop-shadow-[0_2px_8px_rgba(255,153,51,0.25)]">
                OFFICIAL DRAFTER
              </h1>
              <p className="text-[10px] text-amber-400 font-semibold tracking-wider font-space uppercase">
                Next-Gen Administrative Drafting Console
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* API key warning indicator */}
            {!settings.apiKey && (
              <button 
                onClick={() => setIsSettingsOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-amber-950/20 border border-amber-900/30 text-amber-400 rounded-lg animate-pulse cursor-pointer"
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                Missing API Key
              </button>
            )}

            {/* Dark Mode Toggle */}
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="p-2 rounded-lg border border-zinc-800 text-zinc-400 hover:bg-zinc-900/50 hover:border-cyan-500/30 hover:text-cyan-400 transition-all cursor-pointer"
              aria-label="Toggle Dark Mode"
            >
              {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-zinc-400" />}
            </button>

            {/* Settings Trigger */}
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium border border-zinc-800 text-zinc-300 rounded-lg hover:bg-zinc-900/50 hover:border-cyan-500/30 hover:text-cyan-400 transition-all cursor-pointer"
              aria-label="Configure AI Settings"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Split-Screen */}
      <main className="flex-1 w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-8 print:p-0 print:m-0 print:block box-border">
        
        {/* Left Panel: Input & Controls (Editor Column) */}
        <section className="editor-container w-full flex flex-col gap-6 print:hidden box-border">
          
          {/* 1. Format Reference */}
          <div className="glass-panel border-vibgyor rounded-xl p-5 glow-card space-y-3">
            <h2 className="text-xs font-bold text-cyan-400 font-space tracking-widest uppercase">
              1. Format Reference
            </h2>
            <ReferenceUpload
              onUploadSuccess={handleUploadSuccess}
              onError={handleParserError}
              currentReference={currentReference}
            />
          </div>

          {/* 2. RAG Section */}
          <div className="glass-panel border-vibgyor rounded-xl p-5 glow-card space-y-3">
            <h2 className="text-xs font-bold text-indigo-400 font-space tracking-widest uppercase">
              2. RAG (Knowledge Reference)
            </h2>
            <RAGSection
              ragDocuments={ragDocuments}
              onUploadRAG={handleUploadRAG}
              onDeleteDocument={handleDeleteRAGDocument}
              isRAGLoading={isRAGLoading}
              provider={settings.provider}
              hasApiKey={!!settings.apiKey}
            />
          </div>

          {/* 3. Metadata & Narrative */}
          <div className="glass-panel rounded-xl p-5 glow-card space-y-3">
            <h2 className="text-xs font-bold text-cyan-400 font-space tracking-widest uppercase">
              3. Metadata & Narrative
            </h2>
            <DocumentForm
              docType={docType}
              setDocType={setDocType}
              formData={formData}
              setFormData={setFormData}
              draftingLanguage={draftingLanguage}
              setDraftingLanguage={setDraftingLanguage}
              toneMode={toneMode}
              setToneMode={setToneMode}
              narrative={narrative}
              setNarrative={setNarrative}
              onSubmit={handleGenerateDocument}
              isLoading={isLoading}
              hasReference={!!currentReference}
            />
          </div>

          {/* Iterative Refinement Chat Panel */}
          <div className="flex-1 min-h-[300px]">
            <ChatPanel
              chatMessages={chatMessages}
              onSendMessage={handleSendChatMessage}
              isLoading={isLoading}
              hasDraft={!!generatedDocument}
            />
          </div>
          
        </section>

        {/* Right Panel: Preview Column */}
        <section className="preview-container w-full glass-panel rounded-xl overflow-hidden glow-card flex flex-col print:border-none print:shadow-none print:bg-white print:p-0 print:m-0 print:block box-border">
          <DocumentPreview
            htmlContent={generatedDocument}
            onContentChange={setGeneratedDocument}
            docType={docType}
            toneMode={toneMode}
            draftingLanguage={draftingLanguage}
          />
        </section>

      </main>

      {/* Settings Modal Dialog */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleSaveSettings}
        currentSettings={settings}
      />

      {/* Notification Toast */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}
