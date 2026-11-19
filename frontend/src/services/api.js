/**
 * Backend API Client for Production Hybrid RAG Application.
 * Communicates with FastAPI backend endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Fetch health and diagnostic status from FastAPI backend.
 * @returns {Promise<Object>} Health check payload
 */
export async function fetchHealthStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Failed to connect to backend health endpoint:', error);
    return {
      success: false,
      error: error.message || 'Unable to connect to FastAPI backend server',
    };
  }
}

/**
 * Fetch document upload configuration (limits, etc.).
 * @returns {Promise<Object>}
 */
export async function fetchDocumentConfig() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/documents/config`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Upload a PDF document for ingestion and chunking with progress reporting.
 * @param {File} file - PDF file object
 * @param {Function} [onProgress] - Optional callback receiving { loaded, total, percent }
 * @returns {Promise<Object>} Upload response with document and chunk metrics
 */
export function uploadDocument(file, onProgress) {
  return new Promise((resolve) => {
    try {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      if (onProgress && xhr.upload) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress({ loaded: event.loaded, total: event.total, percent });
          }
        };
      }

      xhr.open('POST', `${API_BASE_URL}/api/v1/documents/upload`);

      xhr.onload = () => {
        try {
          const data = JSON.parse(xhr.responseText || '{}');
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve({ success: true, data });
          } else {
            resolve({
              success: false,
              error: data.detail || `Upload failed with status ${xhr.status}`,
            });
          }
        } catch (err) {
          resolve({
            success: false,
            error: `Failed to parse response: ${err.message}`,
          });
        }
      };

      xhr.onerror = () => {
        resolve({
          success: false,
          error: 'Network error occurred during document upload',
        });
      };

      xhr.send(formData);
    } catch (error) {
      console.error('Document upload failed:', error);
      resolve({
        success: false,
        error: error.message || 'Failed to upload document',
      });
    }
  });
}

/**
 * List all ingested documents.
 * @returns {Promise<Object>} List of DocumentInfo records
 */
export async function fetchDocuments() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/documents`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch documents: HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Failed to fetch documents:', error);
    return {
      success: false,
      error: error.message || 'Unable to load documents',
    };
  }
}

/**
 * Fetch detailed document info including page diagnostics and chunks.
 * @param {string} documentId
 * @returns {Promise<Object>} DocumentDetailResponse
 */
export async function fetchDocumentDetails(documentId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to load document details: HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Failed to load document details:', error);
    return {
      success: false,
      error: error.message || 'Unable to load document details',
    };
  }
}

/**
 * Delete a document and its stored chunks.
 * @param {string} documentId
 * @returns {Promise<Object>} Deletion status
 */
export async function deleteDocument(documentId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Failed to delete document: HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Failed to delete document:', error);
    return {
      success: false,
      error: error.message || 'Unable to delete document',
    };
  }
}

/**
 * Fetch current operational status of the indexing orchestrator.
 * @returns {Promise<Object>} IndexStatusResponse
 */
export async function fetchIndexStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/index/status`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch index status: HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Failed to fetch index status:', error);
    return {
      success: false,
      error: error.message || 'Unable to load index status',
    };
  }
}

/**
 * Fetch comprehensive stats for ChromaDB and BM25 indices.
 * @returns {Promise<Object>} IndexStatsResponse
 */
export async function fetchIndexStats() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/index/stats`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch index stats: HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('Failed to fetch index stats:', error);
    return {
      success: false,
      error: error.message || 'Unable to load index stats',
    };
  }
}

/**
 * Trigger full rebuild of ChromaDB and BM25 indices from stored documents.
 * @returns {Promise<Object>} IndexRebuildResponse
 */
export async function rebuildIndices() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/index/rebuild`, {
      method: 'POST',
      headers: { 'Accept': 'application/json' },
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Index rebuild failed with HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Failed to rebuild indices:', error);
    return {
      success: false,
      error: error.message || 'Index rebuild failed',
    };
  }
}

/**
 * Execute search query across BM25, Dense, or Hybrid RRF retrieval strategies with optional BGE Cross-Encoder Reranking.
 * @param {Object} params - { query: string, retrievalMethod: string, topK: number, rrfK: number, rerank: boolean, rerankerTopK: number, rerankerBatchSize: number }
 * @returns {Promise<Object>} RetrievalResponse
 */
export async function executeRetrievalSearch({
  query,
  retrievalMethod = 'hybrid',
  topK = 5,
  rrfK = 60,
  rerank = false,
  rerankerTopK = null,
  rerankerBatchSize = 16,
}) {
  try {
    const payload = {
      query,
      retrieval_method: retrievalMethod,
      top_k: Number(topK),
      rrf_k: Number(rrfK),
      rerank: Boolean(rerank),
      reranker_batch_size: Number(rerankerBatchSize),
    };
    if (rerankerTopK) {
      payload.reranker_top_k = Number(rerankerTopK);
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/retrieval/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Retrieval search failed with HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Retrieval search failed:', error);
    return {
      success: false,
      error: error.message || 'Retrieval search failed',
    };
  }
}

/**
 * Execute end-to-end question answering with Google Gemini and source citations.
 * @param {Object} params - { query: string, retrievalMethod: string, topK: number, rerank: boolean, rrfK: number, rerankerTopK: number, temperature: number }
 * @returns {Promise<Object>} QAResponse
 */
export async function askQuestion({
  query,
  retrievalMethod = 'hybrid',
  topK = 5,
  rerank = true,
  rrfK = 60,
  rerankerTopK = 3,
  temperature = 0.2,
}) {
  try {
    const payload = {
      query,
      retrieval_method: retrievalMethod,
      top_k: Number(topK),
      rerank: Boolean(rerank),
      rrf_k: Number(rrfK),
      reranker_top_k: Number(rerankerTopK),
      temperature: Number(temperature),
    };

    const response = await fetch(`${API_BASE_URL}/api/v1/qa/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Question answering failed with HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Question answering failed:', error);
    return {
      success: false,
      error: error.message || 'Question answering failed',
    };
  }
}

/**
 * Retrieve the evaluation benchmark report.
 * @param {string} mode - 'benchmark' (fixed ground-truth baseline) or 'live' (active repository)
 * @returns {Promise<Object>} EvaluationReportResponse
 */
export async function fetchEvaluationResults(mode = 'benchmark') {
  try {
    const url = mode ? `${API_BASE_URL}/api/v1/evaluation/results?mode=${mode}` : `${API_BASE_URL}/api/v1/evaluation/results`;
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Failed to fetch evaluation results: HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Failed to fetch evaluation report:', error);
    return {
      success: false,
      error: error.message || 'Failed to load evaluation benchmark report',
    };
  }
}

/**
 * Trigger a new evaluation benchmark run across 6 configurations.
 * @param {Object} params
 * @returns {Promise<Object>} EvaluationReportResponse
 */
export async function runEvaluation({
  kValues = [1, 3, 5, 8],
  includeAnswers = true,
  useCachedAnswers = true,
  rerunLive = false,
} = {}) {
  try {
    const payload = {
      k_values: kValues,
      include_answer_evaluation: includeAnswers,
      use_cached_answers: useCachedAnswers,
      rerun_live_generation: rerunLive,
    };

    const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Evaluation run failed with HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Evaluation run failed:', error);
    return {
      success: false,
      error: error.message || 'Evaluation run failed',
    };
  }
}

/**
 * Fetch the benchmark dataset including verified status and judgments.
 * @returns {Promise<Object>} EvaluationDataset
 */
export async function fetchEvaluationDataset() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/dataset`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Failed to fetch dataset: HTTP ${response.status}`);
    }

    return { success: true, data };
  } catch (error) {
    console.error('Failed to fetch evaluation dataset:', error);
    return {
      success: false,
      error: error.message || 'Failed to load evaluation dataset',
    };
  }
}

