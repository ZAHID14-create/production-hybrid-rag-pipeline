import React from 'react';
import { Database, Search, ArrowRight, Cpu, FileText, Sparkles } from 'lucide-react';

export default function PipelineOverview() {
  const steps = [
    { title: 'Dense Retrieval', tech: 'ChromaDB + MiniLM', icon: Database },
    { title: 'Keyword Search', tech: 'BM25 Okapi', icon: Search },
    { title: 'Rank Fusion', tech: 'Reciprocal Rank Fusion (RRF)', icon: Cpu },
    { title: 'Re-Ranking', tech: 'BAAI/bge-reranker-base', icon: Cpu },
    { title: 'Synthesis & Citations', tech: 'Google Gemini API', icon: Sparkles },
  ];

  return (
    <div className="pipeline-diagram">
      <h3>
        <Cpu size={20} color="#3b82f6" />
        Hybrid RAG Pipeline Architecture
      </h3>
      <div className="pipeline-steps">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={idx}>
              <div className="pipeline-node">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icon size={16} color="#60a5fa" />
                  <span className="node-title">{step.title}</span>
                </div>
                <span className="node-tech">{step.tech}</span>
              </div>
              {idx < steps.length - 1 && (
                <span className="pipeline-arrow">→</span>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

