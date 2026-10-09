import React, { useState, useEffect } from 'react';
import type { Alert } from '../types';
import { acknowledgeAlert, acknowledgeAllAlerts, getSnapshotUrl, flagAlertFalsePositive, fetchActiveLearningQueue } from '../services/api';
import { Bell, CheckCircle, Image, Check, X, Filter, Flag, AlertTriangle, ShieldCheck } from 'lucide-react';

interface AlertsEvidenceProps {
  alerts: Alert[];
  onRefreshAlerts: () => void;
}

export const AlertsEvidence: React.FC<AlertsEvidenceProps> = ({ alerts, onRefreshAlerts }) => {
  const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null);
  const [selectedAlertId, setSelectedAlertId] = useState<number | null>(null);
  const [selectedAlertTitle, setSelectedAlertTitle] = useState<string>('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [onlyUnack, setOnlyUnack] = useState<boolean>(false);
  const [flaggedIds, setFlaggedIds] = useState<Set<number>>(new Set());
  const [alQueue, setAlQueue] = useState<any[]>([]);
  const [showQueueModal, setShowQueueModal] = useState<boolean>(false);

  useEffect(() => {
    loadAlQueue();
  }, []);

  const loadAlQueue = async () => {
    try {
      const q = await fetchActiveLearningQueue();
      setAlQueue(q);
    } catch (err) {
      console.error('Failed to load active learning queue:', err);
    }
  };

  const handleAcknowledge = async (id: number) => {
    try {
      await acknowledgeAlert(id);
      onRefreshAlerts();
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const handleAcknowledgeAll = async () => {
    try {
      await acknowledgeAllAlerts();
      onRefreshAlerts();
    } catch (err) {
      console.error('Failed to acknowledge all alerts:', err);
    }
  };

  const handleFlagFP = async (id: number) => {
    try {
      await flagAlertFalsePositive(id);
      setFlaggedIds((prev) => new Set([...prev, id]));
      loadAlQueue();
      onRefreshAlerts();
    } catch (err) {
      console.error('Failed to flag false positive:', err);
    }
  };

  const filteredAlerts = alerts.filter((al) => {
    if (onlyUnack && al.acknowledged) return false;
    if (severityFilter !== 'all' && al.severity.toLowerCase() !== severityFilter.toLowerCase()) return false;
    return true;
  });

  const getSeverityBadgeClass = (sev: string) => {
    if (sev === 'Critical') return 'badge-critical';
    if (sev === 'High') return 'badge-high';
    return 'badge-medium';
  };

  const criticalCount = alerts.filter((a) => a.severity === 'Critical' && !a.acknowledged).length;
  const unackCount = alerts.filter((a) => !a.acknowledged).length;
  const ackCount = alerts.filter((a) => a.acknowledged).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Top KPI Summary Grid */}
      <div className="grid-4">
        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: '#fef2f2', color: '#dc2626' }}>
            <AlertTriangle size={22} />
          </div>
          <div>
            <div className="stat-label">Critical Unresolved</div>
            <div className="stat-value">{criticalCount}</div>
            <div style={{ fontSize: '0.7rem', color: '#dc2626', marginTop: 2 }}>
              Requires immediate response
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: '#fffbeb', color: '#d97706' }}>
            <Bell size={22} />
          </div>
          <div>
            <div className="stat-label">Total Active Queue</div>
            <div className="stat-value">{unackCount}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
              Pending review
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: '#f0fdf4', color: '#16a34a' }}>
            <ShieldCheck size={22} />
          </div>
          <div>
            <div className="stat-label">Resolved / Cleared</div>
            <div className="stat-value">{ackCount}</div>
            <div style={{ fontSize: '0.7rem', color: '#16a34a', marginTop: 2 }}>
              Audit logged
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ backgroundColor: 'var(--color-primary-light)', color: 'var(--color-primary)' }}>
            <Flag size={22} />
          </div>
          <div>
            <div className="stat-label">Active Learning Curation</div>
            <div className="stat-value">{alQueue.length}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
              Queued false-positives
            </div>
          </div>
        </div>
      </div>

      {/* Filter and Action Bar */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Bell size={16} />
            <span>High-Priority Incidents & Evidence Feed</span>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button
              onClick={() => setShowQueueModal(true)}
              className="btn btn-secondary btn-sm"
              title="View queued false positives for model retraining"
            >
              <Flag size={13} color="var(--color-primary)" />
              <span>Active Learning Queue ({alQueue.length})</span>
            </button>
            <button
              onClick={handleAcknowledgeAll}
              className="btn btn-secondary btn-sm"
            >
              <CheckCircle size={13} />
              <span>Acknowledge All</span>
            </button>
          </div>
        </div>

        <div className="card-body" style={{ padding: '10px 18px', display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8125rem' }}>
            <Filter size={14} color="var(--color-text-muted)" />
            <span style={{ fontWeight: 500 }}>Severity:</span>
            {['all', 'critical', 'high', 'medium'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`btn btn-sm ${severityFilter === sev ? 'btn-primary' : 'btn-secondary'}`}
                style={{ textTransform: 'capitalize', padding: '3px 9px', fontSize: '0.75rem' }}
              >
                {sev}
              </button>
            ))}
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8125rem', cursor: 'pointer', marginLeft: 'auto' }}>
            <input
              type="checkbox"
              checked={onlyUnack}
              onChange={(e) => setOnlyUnack(e.target.checked)}
              style={{ accentColor: 'var(--color-primary)' }}
            />
            <span>Unacknowledged Only</span>
          </label>
        </div>
      </div>

      {/* Incident Alerts & Evidence Log Table */}
      <div className="card">
        <div className="card-body" style={{ padding: 0 }}>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Severity</th>
                  <th>Origin Camera</th>
                  <th>Track ID</th>
                  <th>Debris Class</th>
                  <th>Incident Message</th>
                  <th>Evidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredAlerts.length > 0 ? (
                  filteredAlerts.map((al) => (
                    <tr key={al.id}>
                      <td>
                        {flaggedIds.has(al.id) ? (
                          <span className="badge badge-medium" style={{ backgroundColor: '#fffbeb', color: '#b45309' }}>
                            Flagged FP
                          </span>
                        ) : al.acknowledged ? (
                          <span className="badge badge-low">Acknowledged</span>
                        ) : (
                          <span className="badge badge-critical">Active Alert</span>
                        )}
                      </td>
                      <td>
                        <span className={`badge ${getSeverityBadgeClass(al.severity)}`}>
                          {al.severity}
                        </span>
                      </td>
                      <td style={{ fontWeight: 600 }}>{al.camera_id.toUpperCase()}</td>
                      <td>#{al.track_id}</td>
                      <td>{al.class_name.replace(/_/g, ' ')}</td>
                      <td style={{ maxWidth: 360, color: 'var(--color-text)' }}>{al.message}</td>
                      <td>
                        {al.snapshot_path ? (
                          <button
                            onClick={() => {
                              setSelectedSnapshot(al.snapshot_path);
                              setSelectedAlertId(al.id);
                              setSelectedAlertTitle(`${al.severity} Alert: ${al.class_name} (Track #${al.track_id})`);
                            }}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                          >
                            <Image size={12} />
                            <span>View Snapshot</span>
                          </button>
                        ) : (
                          <span style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem' }}>No snapshot</span>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 6 }}>
                          {!flaggedIds.has(al.id) && (
                            <button
                              onClick={() => handleFlagFP(al.id)}
                              className="btn btn-secondary btn-sm"
                              title="Flag as False Positive for model retraining curation"
                              style={{ padding: '3px 8px', fontSize: '0.72rem', color: '#b45309' }}
                            >
                              <Flag size={12} />
                              <span>Flag FP</span>
                            </button>
                          )}
                          {!al.acknowledged && (
                            <button
                              onClick={() => handleAcknowledge(al.id)}
                              className="btn btn-secondary btn-sm"
                              title="Mark as acknowledged"
                              style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                            >
                              <Check size={12} />
                              <span>Clear</span>
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={8} style={{ textAlign: 'center', padding: '32px', color: 'var(--color-text-muted)' }}>
                      No alerts match the selected filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Evidence Snapshot Modal */}
      {selectedSnapshot && (
        <div className="modal-backdrop" onClick={() => setSelectedSnapshot(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="card-title">
                <Image size={16} />
                <span>{selectedAlertTitle}</span>
              </div>
              <button
                onClick={() => setSelectedSnapshot(null)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div style={{ backgroundColor: '#0f172a', borderRadius: 'var(--radius-md)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src={getSnapshotUrl(selectedSnapshot)}
                  alt="Incident Snapshot"
                  style={{ maxWidth: '100%', maxHeight: '65vh', objectFit: 'contain' }}
                />
              </div>
              <div style={{ marginTop: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  Path: {selectedSnapshot}
                </span>
                {selectedAlertId && !flaggedIds.has(selectedAlertId) && (
                  <button
                    onClick={() => {
                      handleFlagFP(selectedAlertId);
                      setSelectedSnapshot(null);
                    }}
                    className="btn btn-secondary btn-sm"
                    style={{ color: '#b45309' }}
                  >
                    <Flag size={13} />
                    <span>Flag as False Positive</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Learning Queue Modal */}
      {showQueueModal && (
        <div className="modal-backdrop" onClick={() => setShowQueueModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="card-title">
                <Flag size={16} color="var(--color-primary)" />
                <span>Active Learning Curation Queue ({alQueue.length} Samples)</span>
              </div>
              <button
                onClick={() => setShowQueueModal(false)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: 14 }}>
                These candidate samples were flagged by human operators as optical false positives (e.g. wave foam, reflections, sun glint) and are queued for hard negative mining in the next YOLO training loop.
              </p>
              {alQueue.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
                  {alQueue.map((item, idx) => (
                    <div key={idx} style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)', padding: 10, backgroundColor: 'var(--color-surface-muted)' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text)' }}>
                        Alert #{item.alert_id} - {item.flagged_as}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginTop: 4 }}>
                        Time: {new Date(item.timestamp).toLocaleTimeString()}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>
                        Class: {item.predicted_class}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '24px', color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
                  No false-positive candidates currently queued.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
