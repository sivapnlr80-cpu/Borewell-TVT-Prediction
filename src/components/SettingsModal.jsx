import React, { useState, useEffect } from 'react';
import { X, Save, Key, Eye, EyeOff, ShieldCheck } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, onSave, currentSettings }) {
  const [provider, setProvider] = useState(currentSettings?.provider || 'gemini');
  const [apiKey, setApiKey] = useState(currentSettings?.apiKey || '');
  const [model, setModel] = useState(currentSettings?.model || '');
  const [showKey, setShowKey] = useState(false);

  // Model options mapping
  const modelOptions = {
    gemini: [
      { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (Recommended)' },
      { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (High Accuracy)' },
      { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' }
    ],
    openai: [
      { value: 'gpt-4o', label: 'GPT-4o (Standard)' },
      { value: 'gpt-4o-mini', label: 'GPT-4o-mini (Cost-effective)' }
    ],
    anthropic: [
      { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' }
    ]
  };

  // Sync state when provider changes or modal opens
  useEffect(() => {
    if (isOpen) {
      setProvider(currentSettings?.provider || 'gemini');
      setApiKey(currentSettings?.apiKey || '');
      
      const currentModel = currentSettings?.model;
      const validModels = modelOptions[currentSettings?.provider || 'gemini'];
      // Ensure the model is valid for the provider, otherwise default to first available
      if (currentModel && validModels.some(m => m.value === currentModel)) {
        setModel(currentModel);
      } else {
        setModel(validModels[0].value);
      }
    }
  }, [isOpen, currentSettings]);

  // Handle provider changes by resetting default model
  const handleProviderChange = (newProvider) => {
    setProvider(newProvider);
    setModel(modelOptions[newProvider][0].value);
    
    // Attempt to load provider specific key from localStorage
    const savedKey = localStorage.getItem(`api_key_${newProvider}`) || '';
    setApiKey(savedKey);
  };

  const handleSave = () => {
    // Save key to localstorage specific to the provider so they don't have to keep pasting
    localStorage.setItem(`api_key_${provider}`, apiKey);
    
    onSave({
      provider,
      apiKey,
      model
    });
    onClose();
  };

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-[#030303]/60 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="bg-[#0c0c10]/95 border border-white/5 rounded-xl shadow-2xl w-full max-w-md overflow-hidden transform scale-100 transition-all duration-300">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/5 bg-[#08080a]">
          <div className="flex items-center gap-2">
            <Key className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold font-space tracking-wider text-cyan-400 uppercase">AI Configuration</h3>
          </div>
          <button 
            onClick={onClose} 
            className="p-1 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-zinc-200 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4">
          
          {/* Provider */}
          <div>
            <label className="block text-xs font-semibold tracking-wider text-zinc-455 uppercase mb-1.5">
              LLM Provider
            </label>
            <select
              value={provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              className="w-full cyber-input px-3 py-2 text-sm"
            >
              <option value="gemini">Google Gemini</option>
              <option value="openai">OpenAI (ChatGPT)</option>
              <option value="anthropic">Anthropic Claude</option>
            </select>
          </div>

          {/* Model */}
          <div>
            <label className="block text-xs font-semibold tracking-wider text-zinc-455 uppercase mb-1.5">
              Model Version
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full cyber-input px-3 py-2 text-sm"
            >
              {modelOptions[provider].map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* API Key */}
          <div>
            <label className="block text-xs font-semibold tracking-wider text-zinc-455 uppercase mb-1.5">
              API Key
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={`Enter your ${provider === 'gemini' ? 'Gemini' : provider === 'openai' ? 'OpenAI' : 'Anthropic'} API Key`}
                className="w-full cyber-input pl-3 pr-10 py-2 text-sm font-mono"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-200 cursor-pointer"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="mt-1.5 text-xxs text-zinc-500 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
              Keys are stored locally in your browser's local storage and never sent elsewhere.
            </p>
          </div>

        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t border-white/5 bg-[#08080a]">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium cyber-button-secondary rounded-lg cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium cyber-button-primary rounded-lg cursor-pointer"
          >
            <Save className="w-4 h-4" />
            Save Settings
          </button>
        </div>

      </div>
    </div>
  );
}
