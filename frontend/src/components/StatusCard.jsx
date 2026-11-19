import React from 'react';

export default function StatusCard({ icon: Icon, title, description, tag, tagType = 'blue', statusText }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon" style={{ background: 'rgba(51, 65, 85, 0.5)' }}>
          <Icon size={20} color="#60a5fa" />
        </div>
        <div>
          <h3 className="card-title">{title}</h3>
        </div>
      </div>
      <p className="card-desc">{description}</p>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className={`badge-tag badge-${tagType}`}>{tag}</span>
        {statusText && (
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{statusText}</span>
        )}
      </div>
    </div>
  );
}

