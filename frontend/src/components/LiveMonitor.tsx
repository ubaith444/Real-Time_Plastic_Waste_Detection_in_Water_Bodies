import React, { useState, useEffect, useRef } from 'react';
import type { Camera, StreamTelemetry } from '../types';
import {
  getMjpegStreamUrl,
  getQuadStreamUrl,
  getStationStreamUrl,
  getDirectSnapshotDownloadUrl,
  updateStreamConfig,
  activateCamera,
  toggleSahi,
  updateDroneAltitude,
  updateCameraConfig
} from '../services/api';
import {
  Camera as CameraIcon,
  Sliders,
  Layers,
  LayoutGrid,
  Maximize2,
  Volume2,
  VolumeX,
  Plane,
  Play,
  Square,
  Radio,
  Download,
  Video,
  Edit2,
  CheckCircle,
  X
} from 'lucide-react';

interface LiveMonitorProps {
  cameras: Camera[];
  telemetry: StreamTelemetry | null;
  onCameraSwitched: () => void;
}

export const LiveMonitor: React.FC<LiveMonitorProps> = ({ cameras, telemetry, onCameraSwitched }) => {
  const [viewMode, setViewMode] = useState<'single' | 'quad'>('single');
  const [quadViewType, setQuadViewType] = useState<'cards' | 'composite'>('cards');
  const [confidenceThresh, setConfidenceThresh] = useState<number>(0.40);
  const [iouThresh, setIouThresh] = useState<number>(0.45);
  const [suppressionSec, setSuppressionSec] = useState<number>(15.0);
  const [switching, setSwitching] = useState<boolean>(false);
  const [sahiActive, setSahiActive] = useState<boolean>(telemetry?.sahi_enabled ?? true);
  const [audioAlarmEnabled, setAudioAlarmEnabled] = useState<boolean>(false);
  const [droneAlt, setDroneAlt] = useState<number>(28.5);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [snapshotNotice, setSnapshotNotice] = useState<string | null>(null);

  // Edit Camera Source State
  const [editingCam, setEditingCam] = useState<Camera | null>(null);
  const [editName, setEditName] = useState<string>('');
  const [editSourceUrl, setEditSourceUrl] = useState<string>('');
  const [editLocation, setEditLocation] = useState<string>('');
  const [savingCam, setSavingCam] = useState<boolean>(false);

  const [streamStates, setStreamStates] = useState<Record<string, boolean>>({
    'drone-01': true,
    'boat-02': true,
    'underwater-03': true,
    'webcam-04': true,
  });

  // Synthesized Web Audio Alarm Bell for Critical incidents
  const playAlarmBell = () => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.setValueAtTime(1175, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.45);
    } catch (e) {
      console.warn('Audio alarm bell exception:', e);
    }
  };

  const prevAlertCount = useRef<number>(0);
  useEffect(() => {
    if (audioAlarmEnabled && telemetry) {
      const hasCriticalGhostNet = telemetry.tracks?.some((t) => t.class_name === 'fishing_net_rope') || telemetry.density_badge === 'badge-critical';
      if ((telemetry.recent_alert_count > 0 && telemetry.recent_alert_count !== prevAlertCount.current) || hasCriticalGhostNet) {
        playAlarmBell();
      }
      prevAlertCount.current = telemetry.recent_alert_count;
    }
  }, [telemetry?.recent_alert_count, telemetry?.density_badge, audioAlarmEnabled]);

  const handleAltitudeChange = async (alt: number) => {
    setDroneAlt(alt);
    try {
      await updateDroneAltitude(alt);
    } catch (err) {
      console.error('Failed to update drone altitude:', err);
    }
  };

  const handleCameraChange = async (camId: string) => {
    try {
      setSwitching(true);
      await activateCamera(camId);
      onCameraSwitched();
    } catch (err) {
      console.error('Failed to switch camera:', err);
    } finally {
      setSwitching(false);
    }
  };

  const toggleStream = (camId: string) => {
    setStreamStates((prev) => ({
      ...prev,
      [camId]: !prev[camId],
    }));
  };

  const handleConfigUpdate = async (conf: number, iou: number, supp: number) => {
    setConfidenceThresh(conf);
    setIouThresh(iou);
    setSuppressionSec(supp);
    try {
      await updateStreamConfig({
        confidence_threshold: conf,
        iou_threshold: iou,
        suppression_window_sec: supp,
      });
    } catch (err) {
      console.error('Failed to update config:', err);
    }
  };

  const handleToggleSahi = async () => {
    const newState = !sahiActive;
    setSahiActive(newState);
    try {
      await toggleSahi(newState);
    } catch (err) {
      console.error('Failed to toggle SAHI:', err);
    }
  };

  const handleSnapshotDownload = () => {
    const link = document.createElement('a');
    link.href = getDirectSnapshotDownloadUrl();
    link.download = `marine_debris_snapshot_${Date.now()}.jpg`;
    link.click();
    setSnapshotNotice('Snapshot downloaded to disk');
    setTimeout(() => setSnapshotNotice(null), 3000);
  };

  const openEditModal = (cam: Camera) => {
    setEditingCam(cam);
    setEditName(cam.name);
    setEditSourceUrl(cam.source_url);
    setEditLocation(cam.location_name);
  };

  const handleSaveCameraConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingCam) return;
    try {
      setSavingCam(true);
      await updateCameraConfig(editingCam.id, {
        name: editName,
        source_url: editSourceUrl,
        location_name: editLocation,
      });
      setEditingCam(null);
      onCameraSwitched();
    } catch (err) {
      console.error('Failed to save camera config:', err);
    } finally {
      setSavingCam(false);
    }
  };

  const activeCam = cameras.find((c) => c.is_active) || cameras[0];
  const activeStreamUrl = getMjpegStreamUrl();
  const quadStreamUrl = getQuadStreamUrl();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* 1. Camera Fleet Management & Ingestion Gateway Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <CameraIcon size={16} />
            <span>Camera Fleet & Ingestion Gateway</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="badge badge-low" style={{ fontSize: '0.7rem' }}>
              4 Streams Connected
            </span>
            <span className="badge badge-medium" style={{ fontSize: '0.7rem' }}>
              Threaded Ring-Buffer Active
            </span>
          </div>
        </div>

        <div className="card-body" style={{ padding: 0 }}>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Stream Status</th>
                  <th>Camera Station</th>
                  <th>Platform Type</th>
                  <th>Ingestion Source</th>
                  <th>Deployment Location</th>
                  <th>Resolution & FPS</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((cam) => {
                  const isStreaming = streamStates[cam.id] ?? true;
                  const isActive = cam.is_active;
                  return (
                    <tr key={cam.id} style={{ backgroundColor: isActive ? 'var(--color-primary-light)' : undefined }}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              backgroundColor: isStreaming ? '#16a34a' : '#94a3b8',
                              boxShadow: isStreaming ? '0 0 6px #16a34a' : 'none',
                            }}
                          />
                          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isStreaming ? '#16a34a' : 'var(--color-text-muted)' }}>
                            {isStreaming ? (isActive ? 'Active HUD' : 'Streaming') : 'Standby'}
                          </span>
                        </div>
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--color-text)' }}>
                        {cam.name}
                      </td>
                      <td>
                        <span className="badge badge-medium" style={{ fontSize: '0.7rem' }}>
                          {cam.environment_type.replace('_', ' ').toUpperCase()}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                        {cam.source_url}
                      </td>
                      <td style={{ color: 'var(--color-text-secondary)' }}>
                        {cam.location_name}
                      </td>
                      <td>
                        <span style={{ fontWeight: 500 }}>{cam.resolution}</span>
                        <span style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem', marginLeft: 4 }}>
                          @{cam.fps} FPS
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <button
                            onClick={() => handleCameraChange(cam.id)}
                            className={`btn btn-sm ${isActive ? 'btn-primary' : 'btn-secondary'}`}
                            style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                            disabled={switching}
                            title="Set as active monitoring station"
                          >
                            <Radio size={12} />
                            <span>{isActive ? 'Current Feed' : 'Select'}</span>
                          </button>
                          <button
                            onClick={() => toggleStream(cam.id)}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                            title={isStreaming ? 'Pause stream' : 'Resume stream'}
                          >
                            {isStreaming ? <Square size={12} color="#dc2626" /> : <Play size={12} color="#16a34a" />}
                            <span>{isStreaming ? 'Stop' : 'Start'}</span>
                          </button>
                          <button
                            onClick={() => openEditModal(cam)}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '0.75rem' }}
                            title="Edit camera source configuration"
                          >
                            <Edit2 size={12} />
                            <span>Edit</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 2. Live Detection Studio & Controls */}
      <div className="layout-monitor">
        {/* Left Column: Video Viewport & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="card">
            {/* Viewport Header Controls */}
            <div className="card-header" style={{ padding: '10px 16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>
                  {viewMode === 'single' ? activeCam?.name : 'Multi-Station Quad Grid (4-Way)'}
                </span>
                <span className="badge badge-low" style={{ fontSize: '0.68rem' }}>
                  {telemetry?.fps ? `${telemetry.fps.toFixed(1)} FPS` : 'LIVE'}
                </span>
                {telemetry?.latency_ms !== undefined && (
                  <span className="badge badge-medium" style={{ fontSize: '0.68rem' }}>
                    {telemetry.latency_ms.toFixed(1)} ms
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                {/* Single / Quad Toggle */}
                <div style={{ display: 'flex', backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-sm)', padding: 2, border: '1px solid var(--color-border)' }}>
                  <button
                    onClick={() => setViewMode('single')}
                    className={`btn btn-sm ${viewMode === 'single' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                  >
                    <Maximize2 size={12} />
                    <span>Single</span>
                  </button>
                  <button
                    onClick={() => setViewMode('quad')}
                    className={`btn btn-sm ${viewMode === 'quad' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                  >
                    <LayoutGrid size={12} />
                    <span>Quad Grid</span>
                  </button>
                </div>

                {viewMode === 'quad' && (
                  <div style={{ display: 'flex', backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-sm)', padding: 2, border: '1px solid var(--color-border)' }}>
                    <button
                      onClick={() => setQuadViewType('cards')}
                      className={`btn btn-sm ${quadViewType === 'cards' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                    >
                      Independent Feeds
                    </button>
                    <button
                      onClick={() => setQuadViewType('composite')}
                      className={`btn btn-sm ${quadViewType === 'composite' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                    >
                      Composite Mosaic
                    </button>
                  </div>
                )}

                {/* Alarm audio bell toggle */}
                <button
                  onClick={() => setAudioAlarmEnabled(!audioAlarmEnabled)}
                  className={`btn btn-sm ${audioAlarmEnabled ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '4px 8px' }}
                  title={audioAlarmEnabled ? 'Audio alarm active' : 'Audio alarm muted'}
                >
                  {audioAlarmEnabled ? <Volume2 size={13} /> : <VolumeX size={13} />}
                </button>
              </div>
            </div>

            {/* Video Canvas Container */}
            <div className="card-body" style={{ padding: 12 }}>
              {viewMode === 'single' ? (
                <div className="video-frame-wrapper">
                  <img
                    src={isPaused ? undefined : activeStreamUrl}
                    alt="Live Detection Feed"
                    style={{ opacity: isPaused ? 0.3 : 1 }}
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  {isPaused && (
                    <div style={{ position: 'absolute', color: '#ffffff', fontWeight: 600, fontSize: '1rem', backgroundColor: 'rgba(0,0,0,0.6)', padding: '6px 14px', borderRadius: 6 }}>
                      STREAM PAUSED
                    </div>
                  )}
                  {isRecording && (
                    <div style={{ position: 'absolute', top: 12, left: 12, display: 'flex', alignItems: 'center', gap: 6, backgroundColor: 'rgba(220, 38, 38, 0.85)', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: '0.7rem', fontWeight: 600 }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#fff', animation: 'pulse 1s infinite' }} />
                      REC
                    </div>
                  )}
                </div>
              ) : quadViewType === 'composite' ? (
                <div className="video-frame-wrapper">
                  <img
                    src={quadStreamUrl}
                    alt="Quad Composite Stream"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                </div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                  {cameras.map((c) => (
                    <div key={c.id} style={{ position: 'relative', borderRadius: 'var(--radius-md)', overflow: 'hidden', backgroundColor: '#0f172a', aspectRatio: '16/9', border: c.is_active ? '2px solid var(--color-primary)' : '1px solid var(--color-border)' }}>
                      <img
                        src={getStationStreamUrl(c.id)}
                        alt={c.name}
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                      <div style={{ position: 'absolute', top: 6, left: 6, backgroundColor: 'rgba(15, 23, 42, 0.75)', color: '#fff', padding: '2px 6px', borderRadius: 4, fontSize: '0.68rem', fontWeight: 600 }}>
                        {c.name}
                      </div>
                      <button
                        onClick={() => {
                          handleCameraChange(c.id);
                          setViewMode('single');
                        }}
                        style={{ position: 'absolute', bottom: 6, right: 6, backgroundColor: 'var(--color-primary)', border: 'none', color: '#fff', padding: '2px 6px', borderRadius: 4, fontSize: '0.65rem', cursor: 'pointer', fontWeight: 600 }}
                      >
                        Maximize
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Feed Control Bar: Pause / Snapshot / Record Simulation */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10, flexWrap: 'wrap', gap: 8 }}>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button
                    onClick={() => setIsPaused(!isPaused)}
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: '0.75rem' }}
                  >
                    {isPaused ? <Play size={13} color="#16a34a" /> : <Square size={13} color="#dc2626" />}
                    <span>{isPaused ? 'Resume' : 'Pause'}</span>
                  </button>

                  <button
                    onClick={handleSnapshotDownload}
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: '0.75rem' }}
                    title="Download instant high-resolution frame directly to disk"
                  >
                    <Download size={13} />
                    <span>Save Snapshot</span>
                  </button>

                  <button
                    onClick={() => setIsRecording(!isRecording)}
                    className={`btn btn-sm ${isRecording ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ fontSize: '0.75rem' }}
                  >
                    <Video size={13} color={isRecording ? '#ffffff' : '#dc2626'} />
                    <span>{isRecording ? 'Stop Recording' : 'Record Simulation'}</span>
                  </button>

                  {snapshotNotice && (
                    <span style={{ fontSize: '0.72rem', color: '#16a34a', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle size={13} />
                      {snapshotNotice}
                    </span>
                  )}
                </div>

                {/* Telemetry Status Chips */}
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', padding: '2px 8px', borderRadius: 4, backgroundColor: 'var(--color-surface-muted)', border: '1px solid var(--color-border)' }}>
                    Active Debris: <strong style={{ color: 'var(--color-text)' }}>{telemetry?.tracks?.length ?? 0}</strong>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', padding: '2px 8px', borderRadius: 4, backgroundColor: 'var(--color-surface-muted)', border: '1px solid var(--color-border)' }}>
                    Slick Area: <strong style={{ color: 'var(--color-text)' }}>{telemetry?.slick_area_m2 ? `${telemetry.slick_area_m2.toFixed(1)} m²` : '0.0 m²'}</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Model Inference Tuning Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Tuning Sliders */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Sliders size={16} />
                <span>Detection & Inference Tuning</span>
              </div>
            </div>

            <div className="card-body">
              {/* Confidence Threshold */}
              <div className="slider-group">
                <div className="slider-label">
                  <span>Confidence Threshold</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{Math.round(confidenceThresh * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.05"
                  value={confidenceThresh}
                  onChange={(e) => handleConfigUpdate(parseFloat(e.target.value), iouThresh, suppressionSec)}
                  className="slider-input"
                />
                <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
                  Higher values suppress false alarms from waves and sun glint.
                </div>
              </div>

              {/* NMS IoU Threshold */}
              <div className="slider-group">
                <div className="slider-label">
                  <span>NMS IoU Threshold</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{iouThresh.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={iouThresh}
                  onChange={(e) => handleConfigUpdate(confidenceThresh, parseFloat(e.target.value), suppressionSec)}
                  className="slider-input"
                />
                <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
                  Controls overlap deduplication for adjacent floating clusters.
                </div>
              </div>

              {/* Drone Altitude / GSD Slider */}
              <div className="slider-group">
                <div className="slider-label">
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Plane size={12} />
                    <span>Drone Altitude (GSD Scaling)</span>
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{droneAlt.toFixed(1)} m</span>
                </div>
                <input
                  type="range"
                  min="10.0"
                  max="60.0"
                  step="1.0"
                  value={droneAlt}
                  onChange={(e) => handleAltitudeChange(parseFloat(e.target.value))}
                  className="slider-input"
                />
                <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
                  Calculated GSD: {(droneAlt * 0.042).toFixed(2)} cm/px for sub-pixel debris tracking.
                </div>
              </div>

              {/* Batched SAHI Toggle */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--color-surface-muted)', border: '1px solid var(--color-border)', marginTop: 6 }}>
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text)' }}>Batched SAHI Slicing</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>High-altitude tile resolution boost</div>
                </div>
                <button
                  onClick={handleToggleSahi}
                  className={`btn btn-sm ${sahiActive ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '3px 10px', fontSize: '0.72rem' }}
                >
                  {sahiActive ? 'Enabled' : 'Disabled'}
                </button>
              </div>
            </div>
          </div>

          {/* Containment Barrier Logistics Card */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Layers size={16} />
                <span>Containment Boom Logistics</span>
              </div>
            </div>
            <div className="card-body">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Estimated Slick Perimeter:</span>
                  <strong style={{ color: 'var(--color-text)' }}>
                    {telemetry?.slick_area_m2 ? `${(Math.sqrt(telemetry.slick_area_m2) * 4).toFixed(1)} m` : '18.4 m'}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Recommended Barrier Boom:</span>
                  <strong style={{ color: 'var(--color-primary)' }}>
                    {telemetry?.slick_area_m2 ? `${(Math.sqrt(telemetry.slick_area_m2) * 5.2).toFixed(1)} m` : '24.0 m'}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Debris Drift Direction:</span>
                  <strong style={{ color: '#16a34a' }}>SE (142°) @ 0.35 m/s</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Edit Camera Source Modal */}
      {editingCam && (
        <div className="modal-backdrop" onClick={() => setEditingCam(null)}>
          <div className="modal-content" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="card-title">
                <Edit2 size={16} />
                <span>Configure Camera Ingestion [{editingCam.id.toUpperCase()}]</span>
              </div>
              <button
                onClick={() => setEditingCam(null)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--color-text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>
            <form onSubmit={handleSaveCameraConfig} className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text)', marginBottom: 4 }}>
                  Station Display Name
                </label>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  style={{ width: '100%', padding: '7px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)', fontSize: '0.8125rem' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text)', marginBottom: 4 }}>
                  Ingestion Source URL / Stream URI
                </label>
                <input
                  type="text"
                  value={editSourceUrl}
                  onChange={(e) => setEditSourceUrl(e.target.value)}
                  placeholder="e.g. rtsp://192.168.1.100:554/live or 0 for USB"
                  style={{ width: '100%', padding: '7px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)', fontSize: '0.8125rem', fontFamily: 'var(--font-mono)' }}
                  required
                />
                <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: 3 }}>
                  Supports RTSP URIs, HTTP streams, USB device index (0, 1), or synthetic generators.
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text)', marginBottom: 4 }}>
                  Deployment Sector Location
                </label>
                <input
                  type="text"
                  value={editLocation}
                  onChange={(e) => setEditLocation(e.target.value)}
                  style={{ width: '100%', padding: '7px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)', fontSize: '0.8125rem' }}
                  required
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 10 }}>
                <button
                  type="button"
                  onClick={() => setEditingCam(null)}
                  className="btn btn-secondary btn-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={savingCam}
                  className="btn btn-primary btn-sm"
                >
                  {savingCam ? 'Saving...' : 'Save Configuration'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
