import React, { useState } from 'react';
import { Search, CheckCircle2 } from 'lucide-react';

export default function DashboardPlaceholder({ documentCount = 0, chunkCount = 0 }) {
  const [query, setQuery] = useState('');

  return (
    <div className="interactive-mockup">
      <div style={{ display: 'inline-flex', padding: '10px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '50%', marginBottom: '12px' }}>
        <Search size={28} color="#60a5fa" />
      </div>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#f8fafc', marginBottom: '8px' }}>
        Phase 2A Complete: Document Ingestion & Chunking Active
      </h2>
      <p style={{ color: '#94a3b8', fontSize: '0.9rem', maxWidth: '640px', margin: '0 auto 18px' }}>
        {documentCount > 0
          ? `${documentCount} document(s) ingested with ${chunkCount} chunks ready. Dual retrieval (ChromaDB vector store + BM25 keyword index) will be activated in Phase 2B.`
          : 'Upload PDF knowledge documents above. Clean text extraction, page-level metadata tracking, and sentence-aware chunking are fully operational.'}
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <input
          type="text"
          className="mockup-input"
          placeholder="Ask a question across ingested documents (Interactive query engine unlocks in Phase 3)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled
        />
      </div>

      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', gap: '16px', fontSize: '0.85rem', color: '#64748b', flexWrap: 'wrap' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <CheckCircle2 size={15} color="#10b981" /> PyMuPDF Page Text Extraction
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <CheckCircle2 size={15} color="#10b981" /> Subword BPE Sentence Chunking
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <CheckCircle2 size={15} color="#10b981" /> SHA-256 Duplicate Protection
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <CheckCircle2 size={15} color="#10b981" /> Persistent Metadata Catalog
        </span>
      </div>
    </div>
  );
}
