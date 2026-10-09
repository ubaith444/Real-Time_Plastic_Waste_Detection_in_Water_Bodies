import React, { useState, useEffect } from 'react';
import type { Camera, Alert, AnalyticsSummary, StreamTelemetry } from './types';
import { fetchCameras, fetchAlerts, fetchAnalyticsSummary } from './services/api';
import { Navbar } from './components/Navbar';
import { LiveMonitor } from './components/LiveMonitor';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { AlertsEvidence } from './components/AlertsEvidence';
import { ModelManagement } from './components/ModelManagement';
import { AlertTriangle, X } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('live');
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [telemetry, setTelemetry] = useState<StreamTelemetry | null>(null);
  const [activeToast, setActiveToast] = useState<Alert | null>(null);

  // Load baseline data
  const loadData = async () => {
    try {
      const [cams, als, anl] = await Promise.all([
        fetchCameras(),
        fetchAlerts(),
        fetchAnalyticsSummary(),
      ]);
      setCameras(cams);
      setAlerts(als);
      setAnalytics(anl);

      // Check for recent critical alert to show toast
      const recentCrit = als.find((a) => a.severity === 'Critical' && !a.acknowledged);
      if (recentCrit && (!activeToast || activeToast.id !== recentCrit.id)) {
        setActiveToast(recentCrit);
      }
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket for telemetry
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: number | undefined;

    const connectWs = () => {
      try {
        ws = new WebSocket('ws://localhost:8000/ws/telemetry');

        ws.onmessage = (event) => {
          try {
            const data: StreamTelemetry = JSON.parse(event.data);
            setTelemetry(data);
          } catch {
            // ignore parse err
          }
        };

        ws.onclose = () => {
          reconnectTimeout = window.setTimeout(connectWs, 3000);
        };

        ws.onerror = () => {
          ws?.close();
        };
      } catch (err) {
        console.error('WebSocket connection error:', err);
        reconnectTimeout = window.setTimeout(connectWs, 3000);
      }
    };

    connectWs();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  const activeCam = cameras.find((c) => c.is_active) || cameras[0];
  const activeCameraName = activeCam?.name || 'Monitoring Station';
  const unackCount = alerts.filter((a) => !a.acknowledged).length;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--color-background)' }}>
      {/* Minimal Header Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        activeCameraName={activeCameraName}
        fps={telemetry?.fps || 30.0}
        unacknowledgedAlertCount={unackCount}
      />

      {/* Critical Alert Banner Toast */}
      {activeToast && (
        <div
          style={{
            backgroundColor: 'var(--color-error-bg)',
            borderBottom: '1px solid var(--color-error-border)',
            padding: '8px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            color: 'var(--color-error)',
            fontSize: '0.8125rem',
            fontWeight: 500,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={16} />
            <span>
              <strong>CRITICAL HAZARD [{activeToast.camera_id.toUpperCase()}]:</strong> {activeToast.message}
            </span>
          </div>
          <button
            onClick={() => setActiveToast(null)}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--color-error)',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Main Content Area */}
      <main className="main-container" style={{ flex: 1 }}>
        {activeTab === 'live' && (
          <LiveMonitor
            cameras={cameras}
            telemetry={telemetry}
            onCameraSwitched={loadData}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsPanel analytics={analytics} />
        )}

        {activeTab === 'alerts' && (
          <AlertsEvidence
            alerts={alerts}
            onRefreshAlerts={loadData}
          />
        )}

        {activeTab === 'models' && (
          <ModelManagement />
        )}
      </main>

      {/* Minimal Footer - No Project Name */}
      <footer
        style={{
          borderTop: '1px solid var(--color-border)',
          backgroundColor: 'var(--color-surface)',
          padding: '10px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.75rem',
          color: 'var(--color-text-muted)',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <div>
          Decoupled Ring Buffer · YOLOv8 Inference · 2D Kalman Filter Tracking · Streaming Analytics
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#16a34a' }} />
          <span>System Normal</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
