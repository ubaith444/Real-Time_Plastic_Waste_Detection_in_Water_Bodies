import React from 'react';
import {
  Radio,
  BarChart2,
  Bell,
  Cpu,
  Layers,
  Camera
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  activeCameraName: string;
  fps: number;
  unacknowledgedAlertCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  activeCameraName,
  fps,
  unacknowledgedAlertCount,
}) => {
  const tabs = [
    { id: 'live', label: 'Live Studio', icon: Radio },
    { id: 'analytics', label: 'Analytics & Reports', icon: BarChart2 },
    { id: 'alerts', label: 'Alerts & Notifications', icon: Bell, badge: unacknowledgedAlertCount },
    { id: 'models', label: 'AI Model', icon: Cpu },
  ];

  return (
    <header className="app-header">
      <div className="header-container" style={{ maxWidth: '100%', padding: '10px 24px' }}>
        {/* Minimal Icon Symbol - No Project Name */}
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <Layers size={18} />
          </div>
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-text)', letterSpacing: '-0.01em' }}>
            Live Surveillance
          </span>
        </div>

        {/* 4 Core Navigation Tabs */}
        <nav className="header-nav">
          {tabs.map((tab) => {
            const IconComponent = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`nav-button ${isActive ? 'active' : ''}`}
              >
                <IconComponent size={14} />
                <span>{tab.label}</span>
                {tab.badge !== undefined && tab.badge > 0 && (
                  <span className="badge badge-critical" style={{ marginLeft: 2, padding: '1px 5px', fontSize: '0.65rem' }}>
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Active Station & Live Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--color-surface-muted)',
              border: '1px solid var(--color-border)',
              fontSize: '0.75rem',
              color: 'var(--color-text-secondary)',
              fontWeight: 500
            }}
          >
            <Camera size={13} color="var(--color-primary)" />
            <span style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {activeCameraName}
            </span>
          </div>

          <div className="status-badge live" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
            <span className="status-indicator-dot" />
            <span>{fps > 0 ? `${fps.toFixed(1)} FPS` : 'LIVE'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
