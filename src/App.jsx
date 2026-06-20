import React, { useState, useEffect } from 'react';
import { Settings, Moon, Sun, FileText, AlertTriangle, HelpCircle } from 'lucide-react';
import ReferenceUpload from './components/ReferenceUpload';
import DocumentForm from './components/DocumentForm';
import DocumentPreview from './components/DocumentPreview';
import ChatPanel from './components/ChatPanel';
import SettingsModal from './components/SettingsModal';
import Toast from './components/Toast';
import { callLLM } from './utils/llm';

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
    officerName: 'John Doe', // Default values from the example scenario
    designation: 'District Cooperative Officer',
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
      const docHtml = await callLLM({
        provider: settings.provider,
        apiKey: settings.apiKey,
        model: settings.model,
        referenceText: currentReference.text,
        docType,
        metadata: formData,
        draftingLanguage,
        toneMode,
        narrative
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
        refinementRequest: messageText
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
            <img src="/logo.png" alt="Official Drafter Logo" className="w-14 h-14 object-contain rounded-xl border border-emerald-500/25 shadow-[0_0_20px_rgba(16,185,129,0.25)] bg-[#07070a] p-0.5" />
            <div>
              <h1 className="text-2xl font-extrabold font-space tracking-tight text-white bg-gradient-to-r from-emerald-400 via-white to-amber-300 bg-clip-text text-transparent drop-shadow-[0_2px_8px_rgba(16,185,129,0.25)]">
                OFFICIAL DRAFTER
              </h1>
              <p className="text-[10px] text-amber-400 font-semibold tracking-wider font-space uppercase">
                Luxury Financial Tech Drafting Suite
              </p>
            </div>
          </div>

          {/* Floating Fintech Elements */}
          <div className="hidden md:flex items-center gap-6 select-none">
            {/* Holographic Trend Graph */}
            <div className="flex items-center gap-2 bg-slate-900/60 border border-emerald-500/20 rounded-full px-3 py-1 text-emerald-400 text-[10px] font-space tracking-wide shadow-[0_0_12px_rgba(16,185,129,0.15)]">
              <svg className="w-10 h-6 text-emerald-400 filter drop-shadow-[0_0_4px_rgba(16,185,129,0.5)]" viewBox="0 0 40 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M2 18 L10 12 L18 15 L28 4 L38 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="38" cy="8" r="2" fill="#fbbf24" className="animate-ping" />
              </svg>
              <span className="font-semibold">INDEX: +34.8%</span>
            </div>
            
            {/* Translucent Coin */}
            <div className="flex items-center gap-2 bg-slate-900/60 border border-amber-500/20 rounded-full px-3 py-1 text-amber-400 text-[10px] font-space tracking-wide shadow-[0_0_12px_rgba(245,158,11,0.15)]">
              <svg className="w-5 h-5 text-amber-400 animate-spin" style={{ animationDuration: '6s' }} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" strokeDasharray="4 2" />
                <path d="M12 7 V17 M9 10 H15 M9 14 H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span className="font-semibold">STABILITY: DUB-A1</span>
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
      <main className="flex-1 max-w-[1650px] w-full mx-auto p-6 flex flex-col lg:flex-row gap-8 overflow-hidden print:p-0 print:m-0 print:block justify-center">
        
        {/* Left Panel: Input & Controls (Centrally framed to minimize cognitive load) */}
        <section className="w-full lg:w-[45%] xl:w-[40%] flex flex-col gap-6 overflow-auto pr-1 pb-4 scrollbar-thin print:hidden max-w-xl mx-auto lg:mx-0">
          
          {/* Reference Upload Box */}
          <div className="glass-panel border-vibgyor rounded-xl p-5 glow-card space-y-3">
            <h2 className="text-xs font-bold text-cyan-400 font-space tracking-widest uppercase">
              1. Format & Style Reference
            </h2>
            <ReferenceUpload
              onUploadSuccess={handleUploadSuccess}
              onError={handleParserError}
              currentReference={currentReference}
            />
          </div>

          {/* Dynamic Input Form */}
          <div className="glass-panel rounded-xl p-5 glow-card space-y-3">
            <h2 className="text-xs font-bold text-cyan-400 font-space tracking-widest uppercase">
              2. Metadata & Narrative
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

        {/* Right Panel: A4 Preview Screen */}
        <section className="flex-1 glass-panel rounded-xl overflow-hidden glow-card flex flex-col print:border-none print:shadow-none print:bg-white print:p-0 print:m-0 print:block">
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
