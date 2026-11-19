import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  Trash2,
  Eye,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  X,
  Layers,
  Hash,
  Clock,
  HardDrive
} from 'lucide-react';
import { uploadDocument, deleteDocument, fetchDocumentDetails, fetchDocumentConfig } from '../services/api';

export default function DocumentManager({ documents, onDocumentsChanged }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [uploadConfig, setUploadConfig] = useState({
    max_upload_size_mb: 1024,
    max_upload_size_bytes: 1024 * 1024 * 1024,
  });
  const [uploadProgress, setUploadProgress] = useState(null);
  const fileInputRef = useRef(null);

  // Fetch dynamic upload configuration from backend
  useEffect(() => {
    let isMounted = true;
    fetchDocumentConfig().then((res) => {
      if (isMounted && res.success && res.data) {
        setUploadConfig(res.data);
      }
    });
    return () => {
      isMounted = false;
    };
  }, []);

  // Format file size helper
  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format date helper
  const formatDate = (isoString) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch (e) {
      return isoString;
    }
  };

  const handleFileSelection = async (file) => {
    setErrorMessage('');
    setSuccessMessage('');

    if (!file) return;

    // Client-side extension validation
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage(`Invalid file "${file.name}". Only PDF documents (.pdf) are supported.`);
      return;
    }

    // Client-side size validation against configured limit (default 1 GB)
    const maxBytes = uploadConfig.max_upload_size_bytes || 1024 * 1024 * 1024;
    const maxMb = uploadConfig.max_upload_size_mb || 1024;
    if (file.size > maxBytes) {
      setErrorMessage(
        `File exceeds ${formatBytes(maxBytes)} limit (${(file.size / (1024 * 1024)).toFixed(2)} MB).`
      );
      return;
    }

    setUploading(true);
    setUploadProgress({ loaded: 0, total: file.size, percent: 0 });

    const res = await uploadDocument(file, (progress) => {
      setUploadProgress(progress);
    });

    setUploading(false);
    setUploadProgress(null);

    if (res.success) {
      if (res.data.is_duplicate) {
        setSuccessMessage(`Document "${res.data.document.document_name}" was already ingested (found matching SHA-256 hash).`);
      } else {
        setSuccessMessage(`Successfully ingested "${res.data.document.document_name}" into ${res.data.chunks_created} chunks!`);
      }
      onDocumentsChanged();
    } else {
      setErrorMessage(res.error);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleDelete = async (docId, docName) => {
    const confirmed = window.confirm(`Are you sure you want to delete "${docName}"? This will permanently remove the original PDF and all its processed chunks.`);
    if (!confirmed) return;

    setDeletingId(docId);
    const res = await deleteDocument(docId);
    setDeletingId(null);

    if (res.success) {
      setSuccessMessage(`Document "${docName}" deleted successfully.`);
      if (selectedDoc && selectedDoc.document_id === docId) {
        setSelectedDoc(null);
      }
      onDocumentsChanged();
    } else {
      setErrorMessage(res.error);
    }
  };

  const handleViewDetails = async (docId) => {
    setLoadingDetails(true);
    const res = await fetchDocumentDetails(docId);
    setLoadingDetails(false);

    if (res.success) {
      setSelectedDoc(res.data);
    } else {
      setErrorMessage(res.error);
    }
  };

  return (
    <div className="doc-manager-section" style={{ marginBottom: '32px' }}>
      {/* Notifications */}
      {errorMessage && (
        <div className="alert-banner alert-error">
          <AlertCircle size={18} />
          <span style={{ flex: 1 }}>{errorMessage}</span>
          <button className="icon-btn-close" onClick={() => setErrorMessage('')}>
            <X size={16} />
          </button>
        </div>
      )}

      {successMessage && (
        <div className="alert-banner alert-success">
          <CheckCircle2 size={18} />
          <span style={{ flex: 1 }}>{successMessage}</span>
          <button className="icon-btn-close" onClick={() => setSuccessMessage('')}>
            <X size={16} />
          </button>
        </div>
      )}

      {/* Upload Dropzone */}
      <div
        className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && handleFileSelection(e.target.files[0])}
        />

        <div className="dropzone-content">
          <div className="upload-icon-wrapper">
            {uploading ? (
              <RefreshCw size={32} className="spin-animate" color="#3b82f6" />
            ) : (
              <UploadCloud size={32} color="#3b82f6" />
            )}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#f8fafc', marginBottom: '4px' }}>
              {uploading ? 'Processing PDF Document...' : 'Upload Knowledge Base Documents (PDF only)'}
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: uploading && uploadProgress ? '8px' : '0' }}>
              {uploading
                ? uploadProgress && uploadProgress.percent < 100
                  ? `Streaming upload to server (${uploadProgress.percent}% — ${formatBytes(uploadProgress.loaded)} of ${formatBytes(uploadProgress.total)})...`
                  : 'Upload complete! Extracting text page-by-page, normalizing whitespace, and indexing chunks...'
                : `Drag and drop your PDF file here, or click to browse (Max ${formatBytes(uploadConfig.max_upload_size_bytes)}). Password-protected and corrupted files will be flagged.`}
            </p>
            {uploading && uploadProgress && (
              <div style={{ width: '100%', maxWidth: '360px', marginTop: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>
                  <span>{uploadProgress.percent < 100 ? 'Streaming to disk...' : 'Extracting & chunking...'}</span>
                  <span>{uploadProgress.percent}%</span>
                </div>
                <div style={{ width: '100%', height: '6px', backgroundColor: 'rgba(255, 255, 255, 0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${uploadProgress.percent}%`,
                      height: '100%',
                      backgroundColor: uploadProgress.percent < 100 ? '#3b82f6' : '#10b981',
                      transition: 'width 0.2s ease',
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Document Library Table */}
      <div className="card" style={{ marginTop: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={20} color="#3b82f6" />
            <h3 className="card-title" style={{ margin: 0 }}>
              Ingested Document Repository ({documents.length})
            </h3>
          </div>
          <button
            onClick={onDocumentsChanged}
            className="action-btn-secondary"
            title="Refresh repository"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {documents.length === 0 ? (
          <div style={{ padding: '36px', textAlign: 'center', color: '#64748b' }}>
            <FileText size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
            <p style={{ fontSize: '0.95rem', fontWeight: 500 }}>No documents ingested yet.</p>
            <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '4px' }}>
              Upload your first PDF above to begin text extraction, page alignment, and chunking.
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Document Name</th>
                  <th>Size</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>Tokens</th>
                  <th>Uploaded</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.document_id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <FileText size={16} color="#60a5fa" />
                        <span style={{ fontWeight: 500, color: '#f1f5f9' }} title={doc.document_name}>
                          {doc.document_name}
                        </span>
                      </div>
                    </td>
                    <td>{formatBytes(doc.file_size_bytes)}</td>
                    <td>
                      <span className="badge-tag badge-blue" title={`${doc.pages_with_text} pages with text`}>
                        {doc.total_pages} pg{doc.total_pages !== 1 ? 's' : ''}
                      </span>
                    </td>
                    <td>
                      <span className="badge-tag badge-purple">
                        {doc.total_chunks} chunks
                      </span>
                    </td>
                    <td>{doc.total_tokens?.toLocaleString() || '-'}</td>
                    <td style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                      {formatDate(doc.uploaded_at)}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '6px' }}>
                        <button
                          className="action-btn-view"
                          onClick={() => handleViewDetails(doc.document_id)}
                          title="Inspect Chunks"
                        >
                          <Eye size={14} /> Inspect
                        </button>
                        <button
                          className="action-btn-delete"
                          onClick={() => handleDelete(doc.document_id, doc.document_name)}
                          disabled={deletingId === doc.document_id}
                          title="Delete Document"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Chunk Detail Modal */}
      {selectedDoc && (
        <div className="modal-backdrop" onClick={() => setSelectedDoc(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 style={{ fontSize: '1.2rem', color: '#f8fafc', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <FileText size={20} color="#60a5fa" />
                  {selectedDoc.document_name}
                </h3>
                <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '2px' }}>
                  ID: <code style={{ color: '#cbd5e1' }}>{selectedDoc.document_id}</code> | {selectedDoc.total_pages} pages | {selectedDoc.total_chunks} chunks | {selectedDoc.total_tokens?.toLocaleString()} tokens
                </p>
              </div>
              <button className="icon-btn-close" onClick={() => setSelectedDoc(null)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '12px' }}>
                Generated Text Chunks ({selectedDoc.chunks.length})
              </h4>
              <div className="chunk-list">
                {selectedDoc.chunks.map((chunk) => (
                  <div key={chunk.chunk_id} className="chunk-card">
                    <div className="chunk-card-header">
                      <span className="badge-tag badge-cyan">
                        Page {chunk.page_number}
                      </span>
                      <span className="badge-tag badge-purple">
                        Chunk #{chunk.chunk_index}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        {chunk.token_count} tokens | {chunk.character_count} chars
                      </span>
                      <code style={{ fontSize: '0.72rem', color: '#64748b', marginLeft: 'auto' }}>
                        {chunk.chunk_id}
                      </code>
                    </div>
                    <p className="chunk-text">{chunk.chunk_text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
