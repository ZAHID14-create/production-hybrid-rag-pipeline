import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import StatusCard from './components/StatusCard';
import PipelineOverview from './components/PipelineOverview';
import DocumentManager from './components/DocumentManager';
import IndexManager from './components/IndexManager';
import RetrievalComparison from './components/RetrievalComparison';
import QuestionAnswer from './components/QuestionAnswer';
import EvaluationDashboard from './components/EvaluationDashboard';
import { fetchHealthStatus, fetchDocuments } from './services/api';
import { Database, Search, Sparkles, FileText, Layers, Layers3 } from 'lucide-react';

export default function App() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);

  // Load system health
  const refreshHealth = async () => {
    setLoading(true);
    const res = await fetchHealthStatus();
    if (res.success) {
      setHealth(res.data);
    }
    setLoading(false);
  };

  // Load ingested documents repository
  const refreshDocuments = async () => {
    setLoadingDocs(true);
    const res = await fetchDocuments();
    if (res.success) {
      setDocuments(res.data);
    }
    setLoadingDocs(false);
    // Also refresh health to sync document counts
    refreshHealth();
  };

  useEffect(() => {
    refreshHealth();
    refreshDocuments();
  }, []);

  const totalPages = documents.reduce((sum, d) => sum + (d.total_pages || 0), 0);
  const totalChunks = documents.reduce((sum, d) => sum + (d.total_chunks || 0), 0);
  const totalTokens = documents.reduce((sum, d) => sum + (d.total_tokens || 0), 0);

  return (
    <div className="dashboard-container">
      <Header health={health} loading={loading} />

      <PipelineOverview />

      {/* Metrics & Architecture Status Cards */}
      <div className="grid-3">
        <StatusCard
          icon={FileText}
          title="Document Repository"
          description="PyMuPDF multi-page text extraction with 1-indexed source tracking and clean whitespace."
          tag="Phase 2A"
          tagType="blue"
          statusText={`${documents.length} Docs (${totalPages} pgs)`}
        />

        <StatusCard
          icon={Layers3}
          title="Chunking Engine"
          description="Sentence-aware subword BPE token chunking (500 tokens target, 75 overlap)."
          tag="Tokenizer"
          tagType="purple"
          statusText={`${totalChunks} Chunks (${totalTokens.toLocaleString()} tkns)`}
        />

        <StatusCard
          icon={Sparkles}
          title="LLM & Citations"
          description="Google Gemini API generation constrained by cross-encoder reranked context with source citations."
          tag="Gemini API"
          tagType="green"
          statusText={health?.components?.gemini_api || "Key Pending"}
        />
      </div>

      {/* Phase 2A Document Ingestion & Chunk Management */}
      <DocumentManager
        documents={documents}
        onDocumentsChanged={refreshDocuments}
      />

      {/* Phase 2B Dense Vector (ChromaDB) & Lexical (BM25) Index Management */}
      <IndexManager
        onIndexChanged={refreshDocuments}
      />

      {/* Phase 5: Grounded Question Answering with Interactive Source Citations */}
      <QuestionAnswer
        documentCount={documents.length}
        totalChunks={totalChunks}
      />

      {/* Phase 3 & 4: Hybrid Retrieval & Cross-Encoder Reranker Comparison */}
      <RetrievalComparison
        documentCount={documents.length}
        totalChunks={totalChunks}
      />

      {/* Phase 6: RAG Evaluation, Comparison and Performance Benchmarking */}
      <EvaluationDashboard />
    </div>
  );
}
