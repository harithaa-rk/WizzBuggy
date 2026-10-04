# NovaRAG React Frontend

A dark, futuristic React UI for the NovaRAG FastAPI backend.

## Quick Start

```bash
cd novarag-frontend
npm install
npm run dev
```

Then open http://localhost:5173

## Prerequisites
- Your FastAPI backend running at http://127.0.0.1:8000
- Node.js 18+

## File Structure
```
src/
  App.jsx              — root layout, fetches indexed files
  index.css            — full design system (dark theme, Syne + JetBrains Mono)
  main.jsx             — entry point
  utils.js             — Path.basename helper
  components/
    Sidebar.jsx        — indexed files list + clear button
    IngestPanel.jsx    — drag-drop upload, model selector, depth slider
    ChatPanel.jsx      — chat messages, citations, trace, cross-modal links
    CitationItem.jsx   — collapsible citation card (image/audio/text preview)
```

## Features
- Drag-and-drop file ingestion with progress bar
- Model selector: Fast / Smart / Creative
- Retrieval depth slider (3–15)
- Chat with streaming thinking indicator
- Collapsible citation cards with score bar
- Cross-modal link display (text↔audio, text↔image, audio↔image)
- Collapsible traceability map table
- Query term highlighting in snippets
- Sidebar with indexed file list
- Fully dark, grid-background aesthetic (Syne + JetBrains Mono fonts)
