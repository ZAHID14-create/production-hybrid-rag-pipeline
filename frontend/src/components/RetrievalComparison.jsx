import React, { useState } from 'react';
import {
  Search,
  Sparkles,
  Database,
  Layers,
  Clock,
  FileText,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  SlidersHorizontal,
  Bookmark,
  Check,
  ArrowUpDown,
  Zap,
} from 'lucide-react';
import { executeRetrievalSearch } from '../services/api';

export default function RetrievalComparison({ documentCount = 0, totalChunks = 0 }) {
  const [query, setQuery] = useState('');
  const [retrievalMethod, setRetrievalMethod] = useState('hybrid'); // 'hybrid' | 'dense' | 'bm25'
  const [topK, setTopK] = useState(5);
  const [rrfK, setRrfK] = useState(60);
  const [rerank, setRerank] = useState(false);
  const [rerankerTopK, setRerankerTopK] = useState(5);
  const [rerankerBatchSize, setRerankerBatchSize] = useState(16);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState({});

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a search query.');
      return;
    }

    setLoading(true);
    setError(null);

    const res = await executeRetrievalSearch({
      query: query.trim(),
      retrievalMethod,
      topK,
      rrfK,
      rerank,
      rerankerTopK: rerank ? rerankerTopK : null,
      rerankerBatchSize,
    });

    if (res.success) {
      setResponse(res.data);
    } else {
      setError(res.error || 'Failed to execute retrieval search.');
    }
    setLoading(false);
  };

  const toggleChunkExpand = (chunkId) => {
    setExpandedChunks((prev) => ({
      ...prev,
      [chunkId]: !prev[chunkId],
    }));
  };

  const getMethodBadge = (method) => {
    switch (method) {
      case 'hybrid':
        return { label: 'Hybrid RRF Fusion', color: '#818cf8', bg: 'rgba(99, 102, 241, 0.15)' };
      case 'dense':
        return { label: 'Dense Semantic (ChromaDB)', color: '#38bdf8', bg: 'rgba(56, 189, 248, 0.15)' };
      case 'bm25':
        return { label: 'Sparse Lexical (BM25)', color: '#fb923c', bg: 'rgba(251, 146, 60, 0.15)' };
      default:
        return { label: method, color: '#94a3b8', bg: 'rgba(148, 163, 184, 0.15)' };
    }
  };

  return (
    <div className="card" style={{ marginTop: '1.5rem' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '1rem',
          marginBottom: '1.25rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              padding: '0.6rem',
              borderRadius: '8px',
              backgroundColor: 'rgba(139, 92, 246, 0.15)',
              color: '#a78bfa',
              display: 'flex',
            }}
          >
            <Sparkles size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>
              Retrieval & Rank Fusion Comparison
            </h2>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Compare BM25 Keyword Search, Dense ChromaDB Semantic Vectors, and Reciprocal Rank Fusion (RRF)
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="btn btn-secondary"
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}
        >
          <SlidersHorizontal size={14} />
          {showAdvanced ? 'Hide Options' : 'Tune Parameters'}
        </button>
      </div>

      {/* Search Input Form */}
      <form onSubmit={handleSearch} style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div style={{ position: 'relative', flex: '1 1 340px' }}>
            <Search
              size={18}
              style={{
                position: 'absolute',
                left: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-muted)',
              }}
            />
            <input
              type="text"
              placeholder="Ask a technical question across your knowledge documents..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              style={{
                width: '100%',
                padding: '0.75rem 1rem 0.75rem 2.4rem',
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.95rem',
                outline: 'none',
              }}
            />
          </div>

          {/* Strategy Selector Pills */}
          <div
            style={{
              display: 'flex',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              padding: '3px',
              gap: '4px',
            }}
          >
            <button
              type="button"
              onClick={() => setRetrievalMethod('hybrid')}
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '0.5rem 0.85rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.85rem',
                fontWeight: retrievalMethod === 'hybrid' ? 600 : 400,
                backgroundColor: retrievalMethod === 'hybrid' ? '#4f46e5' : 'transparent',
                color: retrievalMethod === 'hybrid' ? '#ffffff' : 'var(--text-secondary)',
                transition: 'all 0.15s ease',
              }}
            >
              <Sparkles size={14} />
              Hybrid (RRF)
            </button>

            <button
              type="button"
              onClick={() => setRetrievalMethod('dense')}
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '0.5rem 0.85rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.85rem',
                fontWeight: retrievalMethod === 'dense' ? 600 : 400,
                backgroundColor: retrievalMethod === 'dense' ? '#0284c7' : 'transparent',
                color: retrievalMethod === 'dense' ? '#ffffff' : 'var(--text-secondary)',
                transition: 'all 0.15s ease',
              }}
            >
              <Database size={14} />
              Dense Vector
            </button>

            <button
              type="button"
              onClick={() => setRetrievalMethod('bm25')}
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '0.5rem 0.85rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.85rem',
                fontWeight: retrievalMethod === 'bm25' ? 600 : 400,
                backgroundColor: retrievalMethod === 'bm25' ? '#ea580c' : 'transparent',
                color: retrievalMethod === 'bm25' ? '#ffffff' : 'var(--text-secondary)',
                transition: 'all 0.15s ease',
              }}
            >
              <Search size={14} />
              BM25 Lexical
            </button>
          </div>

          {/* Cross-Encoder Reranker Toggle */}
          <button
            type="button"
            onClick={() => setRerank(!rerank)}
            disabled={loading}
            title="Toggle BAAI/bge-reranker-base cross-encoder reranking"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0.5rem 0.9rem',
              borderRadius: '8px',
              border: rerank ? '1px solid #10b981' : '1px solid var(--border-color)',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 600,
              backgroundColor: rerank ? 'rgba(16, 185, 129, 0.15)' : 'var(--bg-secondary)',
              color: rerank ? '#34d399' : 'var(--text-secondary)',
              transition: 'all 0.15s ease',
            }}
          >
            <Zap size={15} color={rerank ? '#34d399' : 'currentColor'} />
            <span>Reranking: {rerank ? 'ON' : 'OFF'}</span>
          </button>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.4rem',
              fontWeight: 600,
            }}
          >
            {loading ? (
              <>
                <span className="spin" style={{ display: 'inline-block' }}>⟳</span>
                Searching...
              </>
            ) : (
              <>
                <Search size={16} />
                Retrieve
              </>
            )}
          </button>
        </div>

        {/* Tunable Parameters Panel */}
        {showAdvanced && (
          <div
            style={{
              marginTop: '1rem',
              padding: '1rem',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              display: 'flex',
              gap: '2rem',
              flexWrap: 'wrap',
              alignItems: 'center',
              fontSize: '0.85rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                Candidate Top K:
              </label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{
                  backgroundColor: 'var(--bg-primary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  padding: '0.4rem 0.75rem',
                  fontSize: '0.85rem',
                  outline: 'none',
                }}
              >
                <option value={3}>Top 3</option>
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
                <option value={15}>Top 15</option>
              </select>
            </div>

            {retrievalMethod === 'hybrid' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                  RRF Constant (k):
                </label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={rrfK}
                  onChange={(e) => setRrfK(Number(e.target.value))}
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    padding: '0.4rem 0.75rem',
                    width: '80px',
                    fontSize: '0.85rem',
                    outline: 'none',
                  }}
                />
                <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  (Standard default: 60)
                </span>
              </div>
            )}

            {rerank && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                    Rerank Final Top K:
                  </label>
                  <select
                    value={rerankerTopK}
                    onChange={(e) => setRerankerTopK(Number(e.target.value))}
                    style={{
                      backgroundColor: 'var(--bg-primary)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      padding: '0.4rem 0.75rem',
                      fontSize: '0.85rem',
                      outline: 'none',
                    }}
                  >
                    <option value={3}>Top 3</option>
                    <option value={5}>Top 5</option>
                    <option value={10}>Top 10</option>
                  </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                    Rerank Batch Size:
                  </label>
                  <select
                    value={rerankerBatchSize}
                    onChange={(e) => setRerankerBatchSize(Number(e.target.value))}
                    style={{
                      backgroundColor: 'var(--bg-primary)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      padding: '0.4rem 0.75rem',
                      fontSize: '0.85rem',
                      outline: 'none',
                    }}
                  >
                    <option value={8}>8</option>
                    <option value={16}>16 (Default)</option>
                    <option value={32}>32</option>
                  </select>
                </div>
              </>
            )}
          </div>
        )}
      </form>

      {/* Error Message */}
      {error && (
        <div
          style={{
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '6px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
            color: '#f87171',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Execution Diagnostics Bar */}
      {response && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem',
            padding: '0.75rem 1rem',
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            marginBottom: '1.25rem',
            fontSize: '0.85rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <span
              style={{
                backgroundColor: getMethodBadge(response.retrieval_method).bg,
                color: getMethodBadge(response.retrieval_method).color,
                padding: '0.2rem 0.6rem',
                borderRadius: '4px',
                fontWeight: 600,
                fontSize: '0.75rem',
                textTransform: 'uppercase',
              }}
            >
              {getMethodBadge(response.retrieval_method).label}
            </span>

            {response.rerank_enabled && (
              <span
                style={{
                  backgroundColor: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  color: '#34d399',
                  padding: '0.2rem 0.6rem',
                  borderRadius: '4px',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <Zap size={12} />
                Reranked: {response.reranker_model || 'bge-reranker-base'}
              </span>
            )}

            <span style={{ color: 'var(--text-secondary)' }}>
              Found <strong>{response.total_results}</strong> ranked passage{response.total_results !== 1 ? 's' : ''} for:
            </span>
            <em style={{ color: 'var(--text-primary)' }}>"{response.query}"</em>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', color: 'var(--text-muted)', fontSize: '0.8rem', flexWrap: 'wrap' }}>
            {response.embedding_time_ms !== null && response.embedding_time_ms !== undefined && (
              <span title="Dense query vector inference on CPU">
                Embedding: <strong style={{ color: '#38bdf8' }}>{response.embedding_time_ms} ms</strong>
              </span>
            )}
            {response.reranking_time_ms !== null && response.reranking_time_ms !== undefined && (
              <span title="Cross-encoder sequence classification reranking latency">
                Rerank: <strong style={{ color: '#34d399' }}>{response.reranking_time_ms} ms</strong>
              </span>
            )}
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={13} />
              Total Latency: <strong style={{ color: '#4ade80' }}>{response.execution_time_ms} ms</strong>
            </span>
          </div>
        </div>
      )}

      {/* Retrieved Chunks Results List */}
      {response && response.results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {response.results.map((chunk) => {
            const isExpanded = expandedChunks[chunk.chunk_id];
            const isTruncated = chunk.chunk_text.length > 320;
            const displayText = isExpanded || !isTruncated
              ? chunk.chunk_text
              : `${chunk.chunk_text.slice(0, 320)}...`;

            const hasRerankScore = chunk.reranker_score !== null && chunk.reranker_score !== undefined;
            const rankChanged = chunk.original_rank !== null && chunk.original_rank !== undefined && chunk.original_rank !== chunk.rank;
            const rankDiff = rankChanged ? chunk.original_rank - chunk.rank : 0;

            return (
              <div
                key={chunk.chunk_id}
                style={{
                  backgroundColor: 'var(--bg-secondary)',
                  border: response.rerank_enabled && chunk.rank === 1
                    ? '1px solid rgba(16, 185, 129, 0.4)'
                    : '1px solid var(--border-color)',
                  borderRadius: '8px',
                  padding: '1.1rem',
                  transition: 'border-color 0.15s ease',
                }}
              >
                {/* Chunk Top Metadata Row */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    marginBottom: '0.75rem',
                  }}
                >
                  {/* Rank & Citation Badge */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                    <span
                      style={{
                        backgroundColor: response.rerank_enabled
                          ? (chunk.rank === 1 ? '#059669' : '#0f766e')
                          : (chunk.rank === 1 ? '#4f46e5' : '#334155'),
                        color: '#ffffff',
                        fontWeight: 700,
                        fontSize: '0.8rem',
                        padding: '0.2rem 0.55rem',
                        borderRadius: '4px',
                        minWidth: '28px',
                        textAlign: 'center',
                      }}
                    >
                      #{chunk.rank}
                    </span>

                    {/* Original Rank & Movement Badge if reranked */}
                    {response.rerank_enabled && chunk.original_rank !== null && chunk.original_rank !== undefined && (
                      <span
                        style={{
                          backgroundColor: 'rgba(255, 255, 255, 0.06)',
                          color: 'var(--text-secondary)',
                          fontSize: '0.75rem',
                          padding: '0.2rem 0.45rem',
                          borderRadius: '4px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '3px',
                        }}
                        title={`Initial retrieval rank before reranking: #${chunk.original_rank}`}
                      >
                        was #{chunk.original_rank}
                        {rankDiff > 0 && (
                          <strong style={{ color: '#34d399' }}>▲ +{rankDiff}</strong>
                        )}
                        {rankDiff < 0 && (
                          <strong style={{ color: '#f87171' }}>▼ {rankDiff}</strong>
                        )}
                        {rankDiff === 0 && (
                          <span style={{ color: 'var(--text-muted)' }}>=</span>
                        )}
                      </span>
                    )}

                    {/* Source Citation Pill */}
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        backgroundColor: 'rgba(59, 130, 246, 0.12)',
                        border: '1px solid rgba(59, 130, 246, 0.25)',
                        color: '#93c5fd',
                        borderRadius: '6px',
                        padding: '0.25rem 0.65rem',
                        fontSize: '0.8rem',
                        fontWeight: 500,
                      }}
                    >
                      <Bookmark size={13} />
                      <span>{chunk.document_name}</span>
                      <span style={{ color: '#60a5fa', fontWeight: 600 }}>• Page {chunk.page_number}</span>
                      <span style={{ color: 'var(--text-muted)' }}>(Chunk #{chunk.chunk_index})</span>
                    </div>

                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      ~{chunk.token_count} tokens
                    </span>
                  </div>

                  {/* Score Badges */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                    {/* BGE Reranker Score Badge */}
                    {hasRerankScore && (
                      <span
                        style={{
                          backgroundColor: 'rgba(16, 185, 129, 0.15)',
                          border: '1px solid rgba(16, 185, 129, 0.35)',
                          color: '#34d399',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: 700,
                        }}
                        title="BAAI/bge-reranker-base Sigmoid Relevance Score"
                      >
                        BGE Score: {chunk.reranker_score.toFixed(4)} ({(chunk.reranker_score * 100).toFixed(1)}%)
                      </span>
                    )}

                    {/* Method-specific primary score badge */}
                    {chunk.rrf_score !== null && chunk.rrf_score !== undefined && (
                      <span
                        style={{
                          backgroundColor: 'rgba(99, 102, 241, 0.15)',
                          border: '1px solid rgba(99, 102, 241, 0.3)',
                          color: '#a5b4fc',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                        }}
                        title={`Reciprocal Rank Fusion score = sum(1 / (${response?.rrf_k || 60} + rank))` }
                      >
                        RRF: {chunk.rrf_score.toFixed(6)}
                      </span>
                    )}

                    {chunk.dense_rank !== null && chunk.dense_rank !== undefined ? (
                      <span
                        style={{
                          backgroundColor: 'rgba(56, 189, 248, 0.12)',
                          border: '1px solid rgba(56, 189, 248, 0.25)',
                          color: '#7dd3fc',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                        }}
                        title={`Dense rank #${chunk.dense_rank} with cosine similarity ${chunk.dense_score?.toFixed(4)}`}
                      >
                        Dense: #{chunk.dense_rank} ({chunk.dense_score?.toFixed(3)})
                      </span>
                    ) : (
                      response?.retrieval_method === 'hybrid' && (
                        <span
                          style={{
                            backgroundColor: 'rgba(100, 116, 139, 0.12)',
                            color: '#64748b',
                            padding: '0.2rem 0.45rem',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                          }}
                        >
                          Dense: —
                        </span>
                      )
                    )}

                    {chunk.bm25_rank !== null && chunk.bm25_rank !== undefined ? (
                      <span
                        style={{
                          backgroundColor: 'rgba(251, 146, 60, 0.12)',
                          border: '1px solid rgba(251, 146, 60, 0.25)',
                          color: '#fdba74',
                          padding: '0.2rem 0.55rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                        }}
                        title={`BM25 rank #${chunk.bm25_rank} with lexical score ${chunk.bm25_score?.toFixed(4)}`}
                      >
                        BM25: #{chunk.bm25_rank} ({chunk.bm25_score?.toFixed(3)})
                      </span>
                    ) : (
                      response?.retrieval_method === 'hybrid' && (
                        <span
                          style={{
                            backgroundColor: 'rgba(100, 116, 139, 0.12)',
                            color: '#64748b',
                            padding: '0.2rem 0.45rem',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                          }}
                        >
                          BM25: —
                        </span>
                      )
                    )}
                  </div>
                </div>

                {/* Chunk Text Body */}
                <div
                  style={{
                    fontSize: '0.9rem',
                    lineHeight: 1.6,
                    color: 'var(--text-primary)',
                    backgroundColor: 'rgba(15, 23, 42, 0.4)',
                    padding: '0.75rem 1rem',
                    borderRadius: '6px',
                    borderLeft: `3px solid ${
                      response.rerank_enabled
                        ? (chunk.rank === 1 ? '#10b981' : '#0f766e')
                        : (chunk.rank === 1 ? '#818cf8' : '#475569')
                    }`,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {displayText}
                </div>

                {/* Expand / Collapse Button if long text */}
                {isTruncated && (
                  <button
                    type="button"
                    onClick={() => toggleChunkExpand(chunk.chunk_id)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#60a5fa',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      marginTop: '0.5rem',
                      padding: 0,
                    }}
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp size={14} /> Show less
                      </>
                    ) : (
                      <>
                        <ChevronDown size={14} /> View full chunk ({chunk.chunk_text.length} chars)
                      </>
                    )}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* No Results State */}
      {response && response.results.length === 0 && (
        <div
          style={{
            padding: '2.5rem 1rem',
            textAlign: 'center',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px dashed var(--border-color)',
          }}
        >
          <Search size={32} style={{ color: 'var(--text-muted)', marginBottom: '0.75rem' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            No Matching Chunks Found
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto' }}>
            Neither the dense vector space nor the lexical BM25 index returned matching passages for "{response.query}". Try rephrasing your question or using broader keywords.
          </p>
        </div>
      )}

      {/* Initial Empty State before search */}
      {!response && (
        <div
          style={{
            padding: '2rem 1rem',
            textAlign: 'center',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px dashed var(--border-color)',
          }}
        >
          <div style={{ display: 'inline-flex', padding: '10px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '50%', marginBottom: '12px' }}>
            <Search size={28} color="#60a5fa" />
          </div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
            Phase 3 & 4 Active: Hybrid Retrieval & Cross-Encoder Reranker
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '580px', margin: '0 auto 16px' }}>
            {totalChunks > 0
              ? `${documentCount} document(s) with ${totalChunks} chunks indexed across ChromaDB and BM25. Type any query above to compare dense semantic vectors, BM25 exact keywords, Reciprocal Rank Fusion, and BGE cross-encoder reranking.`
              : 'Upload PDF knowledge documents above to populate both ChromaDB and BM25 indices before running queries.'}
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', fontSize: '0.8rem', color: '#64748b', flexWrap: 'wrap' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> 384-Dim all-MiniLM-L6-v2 Embeddings
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> BM25Okapi Lexical Inverted Index
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> Reciprocal Rank Fusion (k=60)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> BAAI/bge-reranker-base Cross-Encoder
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
