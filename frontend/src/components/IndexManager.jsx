import React, { useState, useEffect } from 'react';
import {
  Database,
  Search,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Layers,
  HardDrive,
  Activity,
} from 'lucide-react';
import { fetchIndexStatus, fetchIndexStats, rebuildIndices } from '../services/api';

export default function IndexManager({ onIndexChanged }) {
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [rebuilding, setRebuilding] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const loadIndexData = async () => {
    setLoading(true);
    const [statsRes, statusRes] = await Promise.all([
      fetchIndexStats(),
      fetchIndexStatus(),
    ]);

    if (statsRes.success) {
      setStats(statsRes.data);
    }
    if (statusRes.success) {
      setStatus(statusRes.data);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadIndexData();
  }, []);

  const handleRebuild = async () => {
    setRebuilding(true);
    setMessage(null);
    setError(null);

    const res = await rebuildIndices();
    if (res.success) {
      setMessage(
        `✓ Rebuild completed: ${res.data.chunks_indexed} chunks indexed from ${res.data.documents_processed} documents in ${res.data.duration_seconds}s.`
      );
      await loadIndexData();
      if (onIndexChanged) onIndexChanged();
    } else {
      setError(res.error || 'Failed to rebuild indices.');
    }
    setRebuilding(false);
  };

  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <div className="card" style={{ marginTop: '1.5rem' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '1rem',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              padding: '0.6rem',
              borderRadius: '8px',
              backgroundColor: 'rgba(99, 102, 241, 0.15)',
              color: '#818cf8',
              display: 'flex',
            }}
          >
            <Database size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>
              Dual-Index Architecture Management
            </h2>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Dense Vector Store (ChromaDB) & Lexical Inverted Index (BM25Okapi)
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={loadIndexData}
            disabled={loading || rebuilding}
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}
            title="Refresh index statistics"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            Refresh
          </button>

          <button
            onClick={handleRebuild}
            disabled={rebuilding || loading}
            className="btn btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.85rem',
              backgroundColor: rebuilding ? '#4f46e5' : undefined,
            }}
          >
            <RefreshCw size={14} className={rebuilding ? 'spin' : ''} />
            {rebuilding ? 'Rebuilding Indices...' : 'Rebuild All Indices'}
          </button>
        </div>
      </div>

      {/* Notifications */}
      {message && (
        <div
          style={{
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '6px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
            color: '#4ade80',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <CheckCircle2 size={16} />
          <span>{message}</span>
        </div>
      )}

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

      {/* Index Metrics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem',
          marginBottom: '1.25rem',
        }}
      >
        {/* ChromaDB Card */}
        <div
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Database size={18} style={{ color: '#38bdf8' }} />
              <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>ChromaDB Dense Vectors</span>
            </div>
            <span
              className={`tag tag-${stats?.chroma_status === 'ready' ? 'green' : 'blue'}`}
              style={{ fontSize: '0.75rem', textTransform: 'uppercase' }}
            >
              {stats?.chroma_status || 'Checking'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Collection Name:</span>
              <code style={{ fontSize: '0.8rem', color: '#38bdf8' }}>{stats?.chroma_collection_name || 'rag_chunks'}</code>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Indexed Chunks:</span>
              <strong style={{ color: 'var(--text-primary)' }}>{stats?.chroma_chunk_count ?? 0} vectors</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Vector Dimension:</span>
              <span>{stats?.embedding_dimension || 384} (Cosine)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Embedding Model:</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-primary)', maxWidth: '170px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={stats?.embedding_model_name}>
                {stats?.embedding_model_name?.split('/').pop() || 'all-MiniLM-L6-v2'}
              </span>
            </div>
          </div>
        </div>

        {/* BM25 Card */}
        <div
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Search size={18} style={{ color: '#fb923c' }} />
              <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>BM25 Lexical Index</span>
            </div>
            <span
              className={`tag tag-${stats?.bm25_status === 'ready' ? 'green' : 'blue'}`}
              style={{ fontSize: '0.75rem', textTransform: 'uppercase' }}
            >
              {stats?.bm25_status || 'Checking'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Algorithm:</span>
              <code style={{ fontSize: '0.8rem', color: '#fb923c' }}>BM25Okapi</code>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Indexed Chunks:</span>
              <strong style={{ color: 'var(--text-primary)' }}>{stats?.bm25_chunk_count ?? 0} docs</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Persistent Corpus:</span>
              <span>{formatBytes(stats?.bm25_corpus_size_bytes)} (JSON)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Pickle Security:</span>
              <span style={{ color: '#4ade80', fontSize: '0.8rem' }}>100% Safe JSON (No Pickles)</span>
            </div>
          </div>
        </div>

        {/* Synchronization Card */}
        <div
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={18} style={{ color: '#a78bfa' }} />
              <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Pipeline Synchronization</span>
            </div>
            <span
              className={`tag tag-${stats?.is_synchronized ? 'green' : 'red'}`}
              style={{ fontSize: '0.75rem' }}
            >
              {stats?.is_synchronized ? 'Synchronized' : 'Desynchronized'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Total Unique Chunks:</span>
              <strong style={{ color: 'var(--text-primary)' }}>{stats?.total_indexed_chunks ?? 0}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Orchestrator State:</span>
              <span style={{ textTransform: 'capitalize' }}>{status?.status || 'idle'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Last Ingestion:</span>
              <span style={{ fontSize: '0.75rem' }}>
                {status?.last_indexed_at
                  ? new Date(status.last_indexed_at).toLocaleTimeString()
                  : 'Pending'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Last Action:</span>
              <span
                style={{
                  fontSize: '0.75rem',
                  maxWidth: '170px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={status?.last_action}
              >
                {status?.last_action || 'None'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
