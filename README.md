# Official Drafter - Government Document Generator

An interactive, AI-powered React web application designed for administrative professionals. The application extracts layout, formatting, and structural sequences from official reference files (PDF/DOCX) and drafts formal letters, memorandums, or proceedings based on plain-language narrative inputs in English or Telugu.

## Key Features

- 📄 **PDF & DOCX Parsing**: Extracts style, layouts, and headers locally inside the browser using PDF.js and Mammoth.js.
- ✍️ **Interactive WYSIWYG Editor**: Simulates an A4 sheet with standard margin layout. The canvas is editable (`contentEditable`) and has a text formatting toolbar (Bold, Italic, lists, alignment) for on-the-fly modifications.
- 🗣️ **Refinement Loop**: An integrated chat interface that allows users to converse with the AI model to refine, adjust, or translate the document in real-time.
- 🎭 **Tone & Style Toggle**:
  - **Strict Official**: Stern, traditional government language.
  - **Modern Professional**: Polite, direct, and constructive.
  - **Empathetic / Friendly**: Warm, supportive, and cooperative.
- 🌐 **Bilingual Drafting**: Full support for English and Telugu (తెలుగు) document drafting, using standard administrative phrasing.
- 💾 **API Key Privacy**: Direct configuration dashboard for Google Gemini, OpenAI, and Anthropic (Claude). API Keys are stored locally in your browser's `localStorage` and never uploaded to external servers.
- 📤 **High-Fidelity Exports**:
  - **PDF Export**: Invokes the browser's native print layout, utilizing media queries to print/save only the A4 preview document.
  - **Word Export**: Downloads the styled HTML wrapped in Microsoft Office-friendly Word XML styling schemas (.doc), preserving margins, tables, and alignments.

---

## Technology Stack

- **Frontend Core**: React 19, JavaScript (ES6+), Vite 8
- **Styling**: Tailwind CSS v4, Custom CSS (A4 Page simulation, PDF print stylesheets)
- **Icons**: Lucide React
- **Document Parsers**: `pdfjs-dist` (using dynamic local web workers), `mammoth`
- **AI Integrations**: Native fetch implementations for Gemini, OpenAI, and Anthropic APIs

---

## Getting Started

### Prerequisites

Ensure you have [Node.js](https://nodejs.org/) installed (v18.x or higher is recommended).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sivapnlr80-cpu/OfficialDrafter.git
   cd OfficialDrafter
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```

### Running Locally

1. Start the local development server:
   ```bash
   npm run dev
   ```
2. Open your web browser and navigate to the local address (usually `http://localhost:5173`).
3. Click the **Settings** gear icon in the top right to configure your API Provider and enter your API Key.
4. Upload a reference file to extract formatting.
5. Fill out the metadata fields (Reference Number, Date, Officer details, or From/To).
6. Enter your narrative, choose your language & tone, and click **Generate Official Document**.

### Building for Production

Compile and bundle the application:
```bash
npm run build
```
This outputs production-ready assets inside the `dist/` directory.
