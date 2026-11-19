import React from 'react';
import { Layers, Activity, Server, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Header({ health, loading }) {
  const isHealthy = health && health.status === 'healthy';

  return (
    <header className="header">
      <div className="header-title-group">
        <div className="logo-badge">
          <Layers size={24} color="#ffffff" />
        </div>
        <div>
          <h1 className="app-title">Production Hybrid RAG Pipeline</h1>
          <p className="app-subtitle">SkillCraft Technology Internship — Task 1: Source Citations & Dual Retrieval</p>
        </div>
      </div>

      <div className="status-badge">
        <Server size={16} color="#94a3b8" />
        <span>Backend API:</span>
        {loading ? (
          <span style={{ color: '#94a3b8' }}>Checking...</span>
        ) : isHealthy ? (
          <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span className="status-indicator healthy"></span>
            Online (v{health.version})
          </span>
        ) : (
          <span style={{ color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span className="status-indicator pending"></span>
            Offline / Standby
          </span>
        )}
      </div>
    </header>
  );
}

