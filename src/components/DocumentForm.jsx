import React from 'react';
import { Sparkles, FileText, Languages, ShieldCheck, HeartHandshake } from 'lucide-react';

export default function DocumentForm({
  docType,
  setDocType,
  formData,
  setFormData,
  draftingLanguage,
  setDraftingLanguage,
  toneMode,
  setToneMode,
  narrative,
  setNarrative,
  onSubmit,
  isLoading,
  hasReference
}) {
  
  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const isFormValid = hasReference && narrative.trim().length > 0;

  return (
    <div className="space-y-4">
      {/* Document Type Selection */}
      <div>
        <label className="block text-[14px] font-bold font-space tracking-wider text-zinc-400 uppercase mb-1.5">
          Document Type
        </label>
        <div className="grid grid-cols-3 gap-2 bg-[#09090c] border border-white/5 p-1 rounded-lg">
          {['Letter', 'Memo', 'Proceedings'].map((type) => {
            const isSelected = docType === type;
            return (
              <button
                key={type}
                type="button"
                onClick={() => setDocType(type)}
                className={`py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/25 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5 border border-transparent'
                }`}
              >
                {type}
              </button>
            );
          })}
        </div>
      </div>

      {/* Dynamic Metadata Fields */}
      <div className="grid grid-cols-2 gap-3">
        {docType === 'Letter' ? (
          <>
            <div>
              <label className="block text-[14px] font-bold text-zinc-400 mb-1">
                From (Sender)
              </label>
              <input
                type="text"
                value={formData.from || ''}
                onChange={(e) => handleInputChange('from', e.target.value)}
                placeholder="e.g. District Collector"
                className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
              />
            </div>
            <div>
              <label className="block text-[14px] font-bold text-zinc-400 mb-1">
                To (Recipient)
              </label>
              <input
                type="text"
                value={formData.to || ''}
                onChange={(e) => handleInputChange('to', e.target.value)}
                placeholder="e.g. Bank Manager, SBI"
                className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
              />
            </div>
          </>
        ) : (
          <>
            <div>
              <label className="block text-[14px] font-bold text-zinc-400 mb-1">
                Officer Name
              </label>
              <input
                type="text"
                value={formData.officerName || ''}
                onChange={(e) => handleInputChange('officerName', e.target.value)}
                placeholder="e.g. John Doe"
                className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
              />
            </div>
            <div>
              <label className="block text-[14px] font-bold text-zinc-400 mb-1">
                Designation
              </label>
              <input
                type="text"
                value={formData.designation || ''}
                onChange={(e) => handleInputChange('designation', e.target.value)}
                placeholder="e.g. District Cooperative Officer"
                className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
              />
            </div>
          </>
        )}
      </div>

      {/* Reference Number & Date Fields */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-[14px] font-bold text-zinc-400 mb-1">
            Rc. No. (Reference Number)
          </label>
          <input
            type="text"
            value={formData.rcNo || ''}
            onChange={(e) => handleInputChange('rcNo', e.target.value)}
            placeholder="e.g. 123/2024/B"
            className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
          />
        </div>
        <div>
          <label className="block text-[14px] font-bold text-zinc-400 mb-1">
            Document Date
          </label>
          <input
            type="text"
            value={formData.docDate || ''}
            onChange={(e) => handleInputChange('docDate', e.target.value)}
            placeholder="e.g. 15.05.2024"
            className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
          />
        </div>
      </div>

      {/* Drafting Language & Tone Mode Selectors */}
      <div className="grid grid-cols-2 gap-3">
        {/* Language Selection */}
        <div>
          <label className="block text-[14px] font-bold text-zinc-400 mb-1.5 flex items-center gap-1">
            <Languages className="w-3.5 h-3.5 text-cyan-400" />
            Language
          </label>
          <select
            value={draftingLanguage}
            onChange={(e) => setDraftingLanguage(e.target.value)}
            className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
          >
            <option value="English">English</option>
            <option value="Telugu">Telugu (తెలుగు)</option>
          </select>
        </div>

        {/* Tone Selection */}
        <div>
          <label className="block text-[14px] font-bold text-zinc-400 mb-1.5">
            Style / Tone
          </label>
          <select
            value={toneMode}
            onChange={(e) => setToneMode(e.target.value)}
            className="w-full cyber-input px-3 py-2 text-[14px] font-bold"
          >
            <option value="Strict Official">Strict Official</option>
            <option value="Modern Professional">Modern Professional</option>
            <option value="Empathetic">Empathetic / Friendly</option>
            <option value="Minimalist Modern">Minimalist Modern</option>
          </select>
        </div>
      </div>

      {/* Tone Description Info box */}
      <div className="p-2.5 rounded-lg border border-white/5 bg-[#09090c]/50 text-xxs text-zinc-400">
        {toneMode === 'Strict Official' && (
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-450 shrink-0" />
            <strong>Strict Official:</strong> Traditional tone matching exact warnings of reference.
          </span>
        )}
        {toneMode === 'Modern Professional' && (
          <span className="flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <strong>Modern Professional:</strong> Polite but firm language. Softens warnings while maintaining authority.
          </span>
        )}
        {toneMode === 'Empathetic' && (
          <span className="flex items-center gap-1.5">
            <HeartHandshake className="w-3.5 h-3.5 text-emerald-450 shrink-0" />
            <strong>Empathetic/Friendly:</strong> Supportive tone focusing on cooperative outcomes.
          </span>
        )}
        {toneMode === 'Minimalist Modern' && (
          <span className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400 shrink-0 animate-pulse" />
            <strong>Minimalist Modern:</strong> Extremely clean, spacing-focused style with modern phrasing.
          </span>
        )}
      </div>

      {/* Narrative Input */}
      <div>
        <label className="block text-[14px] font-bold font-space tracking-wider text-zinc-400 uppercase mb-1.5">
          Situation & Outcome Narrative
        </label>
        <textarea
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
          placeholder="Describe the situation and desired outcome in plain language. e.g. The PACS in Nellore are not updating their EOD. I want them to do it by next Friday or I will take action."
          rows="4"
          className="w-full cyber-input px-3 py-2 text-[14px] font-bold resize-y"
        />
      </div>

      {/* Action Button */}
      <button
        onClick={onSubmit}
        disabled={!isFormValid || isLoading}
        className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 font-semibold text-sm rounded-lg transition-all duration-305 cursor-pointer ${
          isFormValid && !isLoading
            ? 'glossy-navy-blue'
            : 'bg-zinc-900/40 border border-white/5 text-zinc-600 cursor-not-allowed'
        }`}
      >
        <Sparkles className={`w-4 h-4 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
        {isLoading ? 'Drafting Document...' : 'Generate Official Document'}
      </button>

      {!hasReference && (
        <p className="text-center text-xxs text-zinc-500 mt-1">
          * Upload a reference style file above to initiate drafting.
        </p>
      )}
    </div>
  );
}
