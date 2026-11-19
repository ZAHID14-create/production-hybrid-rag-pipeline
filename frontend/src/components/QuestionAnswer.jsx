import React, { useState, useRef } from 'react';
import {
  MessageSquare,
  Sparkles,
  Search,
  Database,
  Layers,
  Zap,
  Clock,
  Bookmark,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  ExternalLink,
  BookOpen,
  SlidersHorizontal,
  Bot,
  FileText,
  Tag,
} from 'lucide-react';
import { askQuestion } from '../services/api';

const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
  'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
  'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing',
  'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
  'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers',
  'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if',
  'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t',
  'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
  'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s',
  'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
  'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re',
  'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
  'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
  'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t',
  'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself',
  'yourselves', 'explain', 'concept', 'describe', 'tell', 'define'
]);

function extractKeywords(queryStr) {
  if (!queryStr) return [];
  const words = queryStr.toLowerCase().replace(/[^a-z0-9\s_-]/g, ' ').split(/\s+/).filter(Boolean);
  const keywords = words.filter((w) => w.length > 1 && !STOP_WORDS.has(w));
  return keywords.length > 0 ? keywords : words.filter((w) => w.length > 1);
}

function extractSectionHeading(chunkText, queryKeywords = []) {
  if (!chunkText) return null;
  const headingRegex = /(?:^|\n)\s*([0-9]+(?:\.[0-9]+)*\s*[-:.)]?\s*[A-Z][A-Za-z0-9\s/&_-]{2,60}:?)/g;
  const matches = [];
  let m;
  while ((m = headingRegex.exec(chunkText)) !== null) {
    const title = m[1].trim();
    if (title.length >= 4 && !/^[0-9\s.:]+$/.test(title)) {
      matches.push({ title, index: m.index });
    }
  }

  if (matches.length === 0) return null;

  if (queryKeywords.length > 0) {
    const lower = chunkText.toLowerCase();
    let firstKeywordIdx = -1;
    for (const kw of queryKeywords) {
      const idx = lower.indexOf(kw.toLowerCase());
      if (idx !== -1 && (firstKeywordIdx === -1 || idx < firstKeywordIdx)) {
        firstKeywordIdx = idx;
      }
    }

    if (firstKeywordIdx !== -1) {
      let bestHeading = null;
      for (const h of matches) {
        if (h.index <= firstKeywordIdx + 40) {
          bestHeading = h.title;
        }
      }
      if (bestHeading) return bestHeading;
    }
  }

  return matches[0].title;
}

function getQueryCenteredSnippet(text, queryStr, targetLen = 320) {
  if (!text) return '';
  if (text.length <= targetLen) return text;

  const keywords = extractKeywords(queryStr);
  if (keywords.length === 0) {
    const slice = text.slice(0, targetLen);
    const lastSpace = slice.lastIndexOf(' ');
    return (lastSpace > targetLen * 0.7 ? slice.slice(0, lastSpace) : slice) + '...';
  }

  const lowerText = text.toLowerCase();
  let bestIdx = -1;

  for (const kw of keywords) {
    const idx = lowerText.indexOf(kw);
    if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
      bestIdx = idx;
    }
  }

  if (bestIdx === -1) {
    const slice = text.slice(0, targetLen);
    const lastSpace = slice.lastIndexOf(' ');
    return (lastSpace > targetLen * 0.7 ? slice.slice(0, lastSpace) : slice) + '...';
  }

  let start = Math.max(0, bestIdx - Math.floor(targetLen * 0.25));
  let end = Math.min(text.length, start + targetLen);

  if (end - start < targetLen && start > 0) {
    start = Math.max(0, end - targetLen);
  }

  if (start > 0) {
    const nextSpace = text.indexOf(' ', start);
    if (nextSpace !== -1 && nextSpace < start + 25) {
      start = nextSpace + 1;
    }
  }

  if (end < text.length) {
    const prevSpace = text.lastIndexOf(' ', end);
    if (prevSpace !== -1 && prevSpace > end - 25) {
      end = prevSpace;
    }
  }

  let snippet = text.slice(start, end).trim();
  if (start > 0) snippet = '... ' + snippet;
  if (end < text.length) snippet = snippet + ' ...';

  return snippet;
}

function renderHighlightedSnippet(snippet, queryKeywords = [], isActivelyCited = false) {
  if (!snippet) return null;
  if (!queryKeywords || queryKeywords.length === 0) {
    return <span>{snippet}</span>;
  }

  const escaped = queryKeywords
    .filter((kw) => kw.length >= 2)
    .map((kw) => kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|');

  if (!escaped) return <span>{snippet}</span>;

  const regex = new RegExp(`(${escaped})`, 'gi');
  const parts = snippet.split(regex);

  return (
    <>
      {parts.map((part, i) => {
        if (regex.test(part)) {
          return (
            <mark
              key={i}
              style={{
                backgroundColor: isActivelyCited ? 'rgba(56, 189, 248, 0.35)' : 'rgba(234, 179, 8, 0.25)',
                color: isActivelyCited ? '#7dd3fc' : '#fef08a',
                padding: '1px 4px',
                borderRadius: '3px',
                fontWeight: 600,
              }}
            >
              {part}
            </mark>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

export default function QuestionAnswer({ documentCount = 0, totalChunks = 0 }) {
  const [query, setQuery] = useState('');
  const [retrievalMethod, setRetrievalMethod] = useState('hybrid');
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [rerankerTopK, setRerankerTopK] = useState(3);
  const [rrfK, setRrfK] = useState(60);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [activeCitation, setActiveCitation] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});

  const sourceRefs = useRef({});

  const handleAsk = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a question to ask.');
      return;
    }

    setLoading(true);
    setLoadingStatus('Querying knowledge base & synthesizing answer with Gemini...');
    setError(null);
    setActiveCitation(null);

    const res = await askQuestion({
      query: query.trim(),
      retrievalMethod,
      topK,
      rerank,
      rrfK,
      rerankerTopK: rerank ? rerankerTopK : topK,
      temperature: 0.2,
    });

    if (res.success) {
      setResponse(res.data);
    } else {
      const errMsg = res.error || '';
      if (errMsg.includes('429') || errMsg.toLowerCase().includes('quota')) {
        setError('Gemini API Quota Exceeded (HTTP 429): Your account quota or rate limit has been reached. Please check your Google AI Studio plan or retry later.');
      } else if (errMsg.includes('401') || errMsg.toLowerCase().includes('authentication') || errMsg.toLowerCase().includes('api key')) {
        setError('Gemini Authentication Error (HTTP 401): Missing or invalid GEMINI_API_KEY. Please verify backend/.env.');
      } else if (errMsg.includes('503') || errMsg.toLowerCase().includes('high demand') || errMsg.toLowerCase().includes('unavailable')) {
        setError('Gemini Service Overloaded (HTTP 503): Both primary and fallback models are currently experiencing high demand. Automatic retries completed. Please retry in a few moments.');
      } else {
        setError(errMsg || 'Failed to generate answer. Check Gemini API key configuration.');
      }
    }
    setLoading(false);
    setLoadingStatus('');
  };

  const handleCitationClick = (citationId) => {
    setActiveCitation(citationId);
    // Expand the source if it is collapsed
    setExpandedSources((prev) => ({ ...prev, [citationId]: true }));

    const el = sourceRefs.current[citationId];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const toggleSourceExpand = (citationId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [citationId]: !prev[citationId],
    }));
  };

  /**
   * Parses the text and converts [1], [2], etc., into interactive, clickable citation pills.
   */
  const renderFormattedAnswer = (text) => {
    if (!text) return null;
    const parts = text.split(/(\[\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const citationId = parseInt(match[1], 10);
        const isActive = activeCitation === citationId;

        return (
          <button
            key={index}
            type="button"
            onClick={() => handleCitationClick(citationId)}
            title={`Jump to Source [${citationId}]`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '0 5px',
              margin: '0 2px',
              borderRadius: '4px',
              border: isActive ? '1px solid #38bdf8' : '1px solid rgba(99, 102, 241, 0.4)',
              backgroundColor: isActive ? 'rgba(56, 189, 248, 0.25)' : 'rgba(99, 102, 241, 0.15)',
              color: isActive ? '#38bdf8' : '#a5b4fc',
              fontSize: '0.78rem',
              fontWeight: 700,
              cursor: 'pointer',
              verticalAlign: 'baseline',
              transform: 'translateY(-1px)',
              transition: 'all 0.15s ease',
            }}
          >
            [{citationId}]
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
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
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              color: '#34d399',
              display: 'flex',
            }}
          >
            <Bot size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>
              Production Q&A with Interactive Citations
            </h2>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Ask questions grounded strictly in retrieved PDF knowledge with verified page citations
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
          {showAdvanced ? 'Hide Pipeline Options' : 'Tune Pipeline Options'}
        </button>
      </div>

      {/* Question Form */}
      <form onSubmit={handleAsk} style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
          <div style={{ position: 'relative', flex: '1 1 360px' }}>
            <MessageSquare
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
              placeholder="Ask anything based on your uploaded PDF knowledge documents..."
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

          {/* Retrieval Method Pills */}
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
                gap: '5px',
                padding: '0.5rem 0.75rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.82rem',
                fontWeight: retrievalMethod === 'hybrid' ? 600 : 400,
                backgroundColor: retrievalMethod === 'hybrid' ? '#4f46e5' : 'transparent',
                color: retrievalMethod === 'hybrid' ? '#ffffff' : 'var(--text-secondary)',
              }}
            >
              <Sparkles size={13} />
              Hybrid (RRF)
            </button>

            <button
              type="button"
              onClick={() => setRetrievalMethod('dense')}
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '0.5rem 0.75rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.82rem',
                fontWeight: retrievalMethod === 'dense' ? 600 : 400,
                backgroundColor: retrievalMethod === 'dense' ? '#0284c7' : 'transparent',
                color: retrievalMethod === 'dense' ? '#ffffff' : 'var(--text-secondary)',
              }}
            >
              <Database size={13} />
              Dense
            </button>

            <button
              type="button"
              onClick={() => setRetrievalMethod('bm25')}
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '0.5rem 0.75rem',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.82rem',
                fontWeight: retrievalMethod === 'bm25' ? 600 : 400,
                backgroundColor: retrievalMethod === 'bm25' ? '#ea580c' : 'transparent',
                color: retrievalMethod === 'bm25' ? '#ffffff' : 'var(--text-secondary)',
              }}
            >
              <Search size={13} />
              BM25
            </button>
          </div>

          {/* Reranker Toggle */}
          <button
            type="button"
            onClick={() => setRerank(!rerank)}
            disabled={loading}
            title="Toggle BAAI/bge-reranker-base cross-encoder reranking"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0.5rem 0.85rem',
              borderRadius: '8px',
              border: rerank ? '1px solid #10b981' : '1px solid var(--border-color)',
              cursor: 'pointer',
              fontSize: '0.82rem',
              fontWeight: 600,
              backgroundColor: rerank ? 'rgba(16, 185, 129, 0.15)' : 'var(--bg-secondary)',
              color: rerank ? '#34d399' : 'var(--text-secondary)',
              transition: 'all 0.15s ease',
            }}
          >
            <Zap size={14} color={rerank ? '#34d399' : 'currentColor'} />
            <span>Rerank: {rerank ? 'ON' : 'OFF'}</span>
          </button>

          {/* Ask Button */}
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.5rem',
              fontWeight: 600,
              backgroundColor: '#10b981',
              borderColor: '#059669',
            }}
          >
            {loading ? (
              <>
                <span className="spin" style={{ display: 'inline-block' }}>⟳</span>
                Synthesizing...
              </>
            ) : (
              <>
                <Sparkles size={16} />
                Ask Gemini
              </>
            )}
          </button>
        </div>

        {/* Pipeline Tuning Panel */}
        {showAdvanced && (
          <div
            style={{
              padding: '1rem',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              display: 'flex',
              gap: '2rem',
              flexWrap: 'wrap',
              alignItems: 'center',
              fontSize: '0.85rem',
              marginBottom: '1rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                Candidate Retrieval Pool (Top K):
              </label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{
                  backgroundColor: 'var(--bg-primary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  padding: '0.35rem 0.65rem',
                  fontSize: '0.85rem',
                  outline: 'none',
                }}
              >
                <option value={3}>3 candidates</option>
                <option value={5}>5 candidates (Default)</option>
                <option value={8}>8 candidates</option>
                <option value={10}>10 candidates</option>
              </select>
            </div>

            {rerank && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                  LLM Context Passages (Reranker Top K):
                </label>
                <select
                  value={rerankerTopK}
                  onChange={(e) => setRerankerTopK(Number(e.target.value))}
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    padding: '0.35rem 0.65rem',
                    fontSize: '0.85rem',
                    outline: 'none',
                  }}
                >
                  <option value={2}>Top 2</option>
                  <option value={3}>Top 3 (Recommended)</option>
                  <option value={5}>Top 5</option>
                </select>
              </div>
            )}

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
                    padding: '0.35rem 0.65rem',
                    width: '70px',
                    fontSize: '0.85rem',
                    outline: 'none',
                  }}
                />
              </div>
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
            padding: '0.85rem 1rem',
            marginBottom: '1.25rem',
            color: '#f87171',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
          }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Answer & Sources Presentation */}
      {response && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Diagnostics Bar */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '0.75rem',
              padding: '0.65rem 1rem',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              fontSize: '0.82rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span
                style={{
                  backgroundColor: response.is_fallback_used ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.15)',
                  color: response.is_fallback_used ? '#fbbf24' : '#34d399',
                  border: response.is_fallback_used ? '1px solid rgba(245, 158, 11, 0.4)' : 'none',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '4px',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                {response.is_fallback_used && <Zap size={11} />}
                Model: {response.model_name} {response.is_fallback_used ? '(Fallback Active)' : ''}
              </span>
              {response.retry_count > 0 && (
                <span
                  style={{
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    color: '#f87171',
                    padding: '0.2rem 0.55rem',
                    borderRadius: '4px',
                    fontWeight: 600,
                    fontSize: '0.75rem',
                  }}
                >
                  Retries: {response.retry_count}
                </span>
              )}
              <span
                style={{
                  backgroundColor: 'rgba(99, 102, 241, 0.15)',
                  color: '#a5b4fc',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '4px',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                }}
              >
                {response.retrieval_method} {response.rerank_enabled ? '+ BGE Rerank' : ''}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-muted)' }}>
              {response.embedding_time_ms && (
                <span>Embed: <strong style={{ color: '#38bdf8' }}>{response.embedding_time_ms} ms</strong></span>
              )}
              <span>Retrieval: <strong style={{ color: '#818cf8' }}>{response.retrieval_time_ms} ms</strong></span>
              {response.reranking_time_ms && (
                <span>Rerank: <strong style={{ color: '#34d399' }}>{response.reranking_time_ms} ms</strong></span>
              )}
              <span>Gemini: <strong style={{ color: '#f59e0b' }}>{response.generation_time_ms} ms</strong></span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                <Clock size={12} />
                Total: <strong style={{ color: '#4ade80' }}>{response.total_execution_time_ms} ms</strong>
              </span>
            </div>
          </div>

          {/* Generated Answer Box */}
          <div
            style={{
              backgroundColor: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              borderRadius: '8px',
              padding: '1.25rem',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Sparkles size={18} color="#818cf8" />
              <h3 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>
                Grounded Answer
              </h3>
              {response.citations.length > 0 && (
                <span
                  style={{
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    color: '#38bdf8',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '12px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                  }}
                >
                  {response.citations.length} Verified Citation{response.citations.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            {response.is_fallback_used && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.65rem',
                  backgroundColor: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.35)',
                  borderRadius: '6px',
                  padding: '0.65rem 0.85rem',
                  marginBottom: '1rem',
                  color: '#fbbf24',
                  fontSize: '0.83rem',
                }}
              >
                <Zap size={16} style={{ flexShrink: 0, color: '#f59e0b' }} />
                <div>
                  <strong>Automatic Model Fallback Active:</strong> Answer was successfully synthesized using{' '}
                  <code style={{ backgroundColor: 'rgba(0, 0, 0, 0.35)', padding: '2px 5px', borderRadius: '4px' }}>
                    {response.model_name}
                  </code>{' '}
                  after primary model experienced high demand.
                  {response.retry_count > 0 && ` (Total retries handled: ${response.retry_count})`}
                  {response.fallback_reason && (
                    <div style={{ fontSize: '0.75rem', color: '#d97706', marginTop: '2px' }}>
                      {response.fallback_reason}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div
              style={{
                fontSize: '0.95rem',
                lineHeight: 1.7,
                color: 'var(--text-primary)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {renderFormattedAnswer(response.answer)}
            </div>
          </div>

          {/* Source Passages Panel */}
          {response.sources.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <BookOpen size={18} color="#38bdf8" />
                <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>
                  Supporting Source Documents & Passages ({response.sources.length})
                </h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {response.sources.map((src) => {
                  const isReferenced = response.citations.includes(src.citation_id);
                  const isHighlighted = activeCitation === src.citation_id;
                  const isExpanded = expandedSources[src.citation_id];
                  const queryKeywords = extractKeywords(query);
                  const detectedHeading = extractSectionHeading(src.chunk_text, queryKeywords);
                  const isLong = src.chunk_text.length > 340;
                  const snippetText = getQueryCenteredSnippet(src.chunk_text, query, 340);
                  const activeText = isExpanded ? src.chunk_text : snippetText;

                  return (
                    <div
                      key={src.citation_id}
                      ref={(el) => (sourceRefs.current[src.citation_id] = el)}
                      style={{
                        backgroundColor: 'var(--bg-secondary)',
                        border: isHighlighted
                          ? '2px solid #38bdf8'
                          : isReferenced
                          ? '1px solid rgba(99, 102, 241, 0.4)'
                          : '1px solid var(--border-color)',
                        borderRadius: '8px',
                        padding: '1rem',
                        transition: 'all 0.2s ease',
                        boxShadow: isHighlighted ? '0 0 16px rgba(56, 189, 248, 0.25)' : 'none',
                      }}
                    >
                      {/* Top Source Header */}
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          flexWrap: 'wrap',
                          gap: '0.5rem',
                          marginBottom: '0.5rem',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                          <span
                            style={{
                              backgroundColor: isReferenced ? '#4f46e5' : '#334155',
                              color: '#ffffff',
                              fontWeight: 700,
                              fontSize: '0.8rem',
                              padding: '0.2rem 0.55rem',
                              borderRadius: '4px',
                            }}
                          >
                            Source [{src.citation_id}]
                          </span>

                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              backgroundColor: 'rgba(59, 130, 246, 0.12)',
                              border: '1px solid rgba(59, 130, 246, 0.25)',
                              color: '#93c5fd',
                              borderRadius: '6px',
                              padding: '0.2rem 0.6rem',
                              fontSize: '0.8rem',
                              fontWeight: 500,
                            }}
                          >
                            <Bookmark size={13} />
                            <span>{src.document_name}</span>
                            <span style={{ color: '#60a5fa', fontWeight: 600 }}>• Page {src.page_number}</span>
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>({src.chunk_id})</span>
                          </div>
                        </div>

                        {/* Scores */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem' }}>
                          {src.reranker_score !== null && src.reranker_score !== undefined && (
                            <span
                              style={{
                                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                                color: '#34d399',
                                padding: '0.15rem 0.45rem',
                                borderRadius: '4px',
                                fontWeight: 600,
                              }}
                            >
                              BGE: {(src.reranker_score * 100).toFixed(1)}%
                            </span>
                          )}
                          {src.rrf_score !== null && src.rrf_score !== undefined && (
                            <span
                              style={{
                                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                                color: '#a5b4fc',
                                padding: '0.15rem 0.45rem',
                                borderRadius: '4px',
                              }}
                            >
                              RRF: {src.rrf_score.toFixed(5)}
                            </span>
                          )}
                          <span style={{ color: 'var(--text-muted)' }}>
                            Rank: #{src.rerank_rank || src.retrieval_rank}
                          </span>
                        </div>
                      </div>

                      {/* Active Citation Badge & Section Heading */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                        {isHighlighted && (
                          <span
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              backgroundColor: 'rgba(56, 189, 248, 0.2)',
                              border: '1px solid rgba(56, 189, 248, 0.4)',
                              borderRadius: '4px',
                              padding: '0.15rem 0.5rem',
                              fontSize: '0.73rem',
                              color: '#38bdf8',
                              fontWeight: 700,
                            }}
                          >
                            <Check size={12} />
                            Actively Supporting Citation for [{src.citation_id}]
                          </span>
                        )}

                        {detectedHeading && (
                          <span
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              backgroundColor: 'rgba(99, 102, 241, 0.15)',
                              border: '1px solid rgba(99, 102, 241, 0.3)',
                              borderRadius: '4px',
                              padding: '0.15rem 0.5rem',
                              fontSize: '0.73rem',
                              color: '#a5b4fc',
                              fontWeight: 600,
                            }}
                          >
                            <Tag size={11} color="#818cf8" />
                            Section: {detectedHeading}
                          </span>
                        )}

                        {!isExpanded && isLong && (
                          <span
                            style={{
                              fontSize: '0.72rem',
                              color: 'var(--text-muted)',
                              fontStyle: 'italic',
                            }}
                          >
                            (Displaying query-focused passage snippet)
                          </span>
                        )}
                      </div>

                      {/* Passage Text */}
                      <div
                        style={{
                          fontSize: '0.88rem',
                          lineHeight: 1.6,
                          color: 'var(--text-primary)',
                          backgroundColor: 'rgba(15, 23, 42, 0.4)',
                          padding: '0.65rem 0.85rem',
                          borderRadius: '6px',
                          borderLeft: `3px solid ${isHighlighted ? '#38bdf8' : isReferenced ? '#818cf8' : '#475569'}`,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {renderHighlightedSnippet(activeText, queryKeywords, isHighlighted)}
                      </div>

                      {/* Expand / Collapse toggle */}
                      {isLong && (
                        <button
                          type="button"
                          onClick={() => toggleSourceExpand(src.citation_id)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#60a5fa',
                            cursor: 'pointer',
                            fontSize: '0.78rem',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            marginTop: '0.45rem',
                            padding: 0,
                          }}
                        >
                          {isExpanded ? (
                            <>
                              <ChevronUp size={13} /> Show focused query snippet
                            </>
                          ) : (
                            <>
                              <ChevronDown size={13} /> View full passage ({src.chunk_text.length} chars)
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Initial Empty State before asking */}
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
          <div
            style={{
              display: 'inline-flex',
              padding: '12px',
              background: 'rgba(16, 185, 129, 0.1)',
              borderRadius: '50%',
              marginBottom: '12px',
            }}
          >
            <Sparkles size={28} color="#34d399" />
          </div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
            Phase 5 Active: Grounded Q&A with Google Gemini
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '580px', margin: '0 auto 16px' }}>
            {totalChunks > 0
              ? `Ready to synthesize answers across ${documentCount} document(s) with ${totalChunks} indexed chunks. Enter any technical question above to test end-to-end RAG with verified inline page citations.`
              : 'Upload PDF knowledge documents above to populate ChromaDB and BM25 before asking questions.'}
          </p>

          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '16px',
              fontSize: '0.8rem',
              color: '#64748b',
              flexWrap: 'wrap',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> Google Gemini API via official SDK
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> Anti-Hallucination Grounding Prompt
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> Verified 1-Based Bracket Citations [1]
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Check size={14} color="#10b981" /> Interactive Click-to-Source Highlights
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
