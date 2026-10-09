import type { Camera, Alert, AnalyticsSummary, MapMarker, UsbDevice } from '../types';

const API_BASE = 'http://localhost:8000';

export async function fetchCameras(): Promise<Camera[]> {
  const res = await fetch(`${API_BASE}/api/cameras`);
  if (!res.ok) throw new Error('Failed to fetch cameras');
  return res.json();
}

export async function scanHardwareCameras(): Promise<UsbDevice[]> {
  const res = await fetch(`${API_BASE}/api/cameras/scan`);
  if (!res.ok) throw new Error('Failed to scan cameras');
  const data = await res.json();
  return data.devices || [];
}

export async function testStreamUrl(url: string): Promise<{ reachable: boolean; resolution?: string; latency_ms?: number; error?: string }> {
  const res = await fetch(`${API_BASE}/api/cameras/test-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error('Failed to test stream');
  return res.json();
}

export async function toggleSahi(enabled: boolean): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/cameras/toggle-sahi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error('Failed to toggle SAHI');
  const data = await res.json();
  return data.sahi_enabled;
}

export async function activateCamera(cameraId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/activate`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to activate camera');
}

export async function fetchAlerts(unacknowledgedOnly = false, severity?: string): Promise<Alert[]> {
  let url = `${API_BASE}/api/alerts?limit=50`;
  if (unacknowledgedOnly) url += '&unacknowledged_only=true';
  if (severity) url += `&severity=${severity}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch alerts');
  return res.json();
}

export async function acknowledgeAlert(alertId: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/alerts/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ acknowledged: true }),
  });
  if (!res.ok) throw new Error('Failed to acknowledge alert');
}

export async function acknowledgeAllAlerts(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/alerts/acknowledge-all`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to acknowledge all alerts');
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  const res = await fetch(`${API_BASE}/api/analytics/summary`);
  if (!res.ok) throw new Error('Failed to fetch analytics summary');
  return res.json();
}

export async function fetchMapMarkers(): Promise<MapMarker[]> {
  const res = await fetch(`${API_BASE}/api/analytics/map-markers`);
  if (!res.ok) throw new Error('Failed to fetch map markers');
  return res.json();
}

export async function updateStreamConfig(config: {
  confidence_threshold?: number;
  iou_threshold?: number;
  suppression_window_sec?: number;
}): Promise<void> {
  const res = await fetch(`${API_BASE}/api/stream/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to update stream configuration');
}

export function getSnapshotUrl(filename: string): string {
  return `${API_BASE}/api/snapshots/${filename}`;
}

export function getMjpegStreamUrl(): string {
  return `${API_BASE}/api/stream/mjpeg`;
}

export function getQuadStreamUrl(): string {
  return `${API_BASE}/api/stream/quad`;
}

export function getStationStreamUrl(envType: string): string {
  return `${API_BASE}/api/stream/station/${envType}`;
}

export async function flagAlertFalsePositive(alertId: number): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/alerts/${alertId}/flag-fp`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to flag alert as false positive');
  return res.json();
}

export async function fetchActiveLearningQueue(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/alerts/active-learning/queue`);
  if (!res.ok) throw new Error('Failed to fetch active learning queue');
  return res.json();
}

export async function updateDroneAltitude(altitudeM: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/stream/drone/altitude`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ altitude_m: altitudeM }),
  });
  if (!res.ok) throw new Error('Failed to update drone altitude');
}

export async function fetchSystemHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/analytics/system-health`);
  if (!res.ok) throw new Error('Failed to fetch system health');
  return res.json();
}

export async function fetchAuditLogs(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/analytics/audit-logs`);
  if (!res.ok) throw new Error('Failed to fetch audit logs');
  return res.json();
}

export async function createAuditLog(payload: { action_type: string; target_entity: string; details: string; user_role?: string }): Promise<void> {
  await fetch(`${API_BASE}/api/analytics/audit-logs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function getCsvReportUrl(): string {
  return `${API_BASE}/api/analytics/export-csv`;
}

export function getDirectSnapshotDownloadUrl(): string {
  return `${API_BASE}/api/stream/snapshot`;
}

export async function fetchModels(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function importModelFile(formData: FormData): Promise<any> {
  const res = await fetch(`${API_BASE}/api/models/import`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to import model' }));
    throw new Error(err.detail || 'Failed to import model');
  }
  return res.json();
}

export async function activateModel(modelName: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelName)}/activate`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to activate model');
  return res.json();
}

export async function updateCameraConfig(cameraId: string, payload: { name?: string; source_url?: string; location_name?: string }): Promise<any> {
  const res = await fetch(`${API_BASE}/api/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to update camera');
  return res.json();
}


