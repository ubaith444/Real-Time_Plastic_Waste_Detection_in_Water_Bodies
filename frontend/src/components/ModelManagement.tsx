import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Layers,
  Zap,
  Award,
  Upload,
  CheckCircle,
  FileCode,
  ArrowRight,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { fetchModels, importModelFile, activateModel } from '../services/api';

export const ModelManagement: React.FC = () => {
  const [selectedRuntime, setSelectedRuntime] = useState<'pytorch' | 'onnx'>('pytorch');
  const [modelsList, setModelsList] = useState<any[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadModels = async () => {
    try {
      const data = await fetchModels();
      setModelsList(data);
    } catch (err) {
      console.error('Failed to load models list:', err);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pt' && ext !== 'onnx') {
      setStatusMsg({ type: 'error', text: 'Unsupported format. Only .pt and .onnx model weights are allowed.' });
      return;
    }

    try {
      setUploading(true);
      setStatusMsg(null);
      const formData = new FormData();
      formData.append('file', file);
      const res = await importModelFile(formData);
      setStatusMsg({ type: 'success', text: res.message || `Successfully imported ${file.name}` });
      await loadModels();
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err.message || 'Failed to import model weights.' });
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleActivate = async (modelName: string) => {
    try {
      setStatusMsg(null);
      const res = await activateModel(modelName);
      setStatusMsg({ type: 'success', text: res.message || `Switched active inference model to ${modelName}` });
      await loadModels();
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err.message || 'Failed to activate model.' });
    }
  };

  const modelClasses = [
    { name: 'plastic_bottle', fillFactor: 0.70, risk: 'Medium', desc: 'Single-use PET beverage containers with buoyancy air pockets' },
    { name: 'plastic_bag', fillFactor: 0.55, risk: 'High', desc: 'Thin film polyethylene bags partially submerged below surface' },
    { name: 'plastic_container', fillFactor: 0.85, risk: 'Medium', desc: 'Rigid HDPE oil/chemical jugs and food packaging' },
    { name: 'fishing_net_rope', fillFactor: 0.40, risk: 'Critical', desc: 'Ghost fishing gear and polypropylene ropes posing marine wildlife choke hazard' },
    { name: 'micro_macro_fragment', fillFactor: 0.65, risk: 'High', desc: 'Degraded hard polymer shards and unclassified floating debris' }
  ];

  const domainBenchmarks = [
    {
      domain: 'Unseen Lake Domain',
      location: 'Dal Lake, FloPWD 2025 Split',
      precision: '84.2%',
      recall: '81.0%',
      map50: '83.4%',
      map5095: '58.1%',
      latency: '18.2 ms',
      notes: 'Dense lily pads, water reflections, drone altitude scale shifts'
    },
    {
      domain: 'Unseen River Domain',
      location: 'Ganges River Sector, FloW Split',
      precision: '79.1%',
      recall: '76.4%',
      map50: '77.2%',
      map5095: '52.3%',
      latency: '18.6 ms',
      notes: 'Brown silt turbidity, floating foam false positives, wave chop'
    },
    {
      domain: 'Unseen Coastal Domain',
      location: 'Mumbai Coastal Harbor, FML Split',
      precision: '81.5%',
      recall: '78.2%',
      map50: '80.1%',
      map5095: '54.7%',
      latency: '18.4 ms',
      notes: 'Intense midday sun glint, salt spray, vessel bow wakes'
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* 1. Header Model Architecture Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Cpu size={16} />
            <span>AI Model Architecture & Inference Engines</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="badge badge-low" style={{ fontSize: '0.7rem' }}>
              Active Model: YOLOv8n-Marine-v1.4
            </span>
            <span className="badge badge-medium" style={{ fontSize: '0.7rem' }}>
              SAHI Batched Forward Pass
            </span>
          </div>
        </div>

        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
            <div style={{ padding: 12, backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Architecture Base</div>
              <div style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--color-text)', marginTop: 4 }}>Ultralytics YOLOv8 Nano</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>3.2M Parameters | 8.7 GFLOPs</div>
            </div>

            <div style={{ padding: 12, backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Inference Resolution</div>
              <div style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--color-text)', marginTop: 4 }}>640 x 640 Native</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>512x512 Slices in Drone Mode</div>
            </div>

            <div style={{ padding: 12, backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Multi-Task Capabilities</div>
              <div style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--color-text)', marginTop: 4 }}>BBox + Polygon Area</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>Debris Convex Hull Envelope</div>
            </div>

            <div style={{ padding: 12, backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Focal Loss Gamma</div>
              <div style={{ fontWeight: 700, fontSize: '0.9375rem', color: '#16a34a', marginTop: 4 }}>fl_gamma = 1.5</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: 2 }}>Class weight boost: 3.0x Ghost Nets</div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Import Model Weights & Available Weights Registry */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Upload size={16} />
            <span>Import Model Weights & Active Engine Registry</span>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button
              onClick={loadModels}
              className="btn btn-secondary btn-sm"
              title="Refresh models"
            >
              <RefreshCw size={12} />
              <span>Refresh</span>
            </button>
            <label className="btn btn-primary btn-sm" style={{ cursor: 'pointer' }}>
              <Upload size={13} />
              <span>{uploading ? 'Importing...' : 'Import .pt / .onnx Model'}</span>
              <input
                type="file"
                accept=".pt,.onnx"
                onChange={handleFileUpload}
                disabled={uploading}
                style={{ display: 'none' }}
              />
            </label>
          </div>
        </div>

        {statusMsg && (
          <div
            style={{
              padding: '8px 18px',
              backgroundColor: statusMsg.type === 'success' ? 'var(--color-success-bg)' : 'var(--color-error-bg)',
              color: statusMsg.type === 'success' ? 'var(--color-success)' : 'var(--color-error)',
              fontSize: '0.78rem',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              borderBottom: '1px solid var(--color-border)'
            }}
          >
            {statusMsg.type === 'success' ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
            <span>{statusMsg.text}</span>
          </div>
        )}

        <div className="card-body" style={{ padding: 0 }}>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Model Weight File</th>
                  <th>Engine Format</th>
                  <th>File Size</th>
                  <th>Date Modified</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {modelsList.length > 0 ? (
                  modelsList.map((m, idx) => (
                    <tr key={idx} style={{ backgroundColor: m.is_active ? 'var(--color-primary-light)' : undefined }}>
                      <td>
                        <span className={`badge ${m.is_active ? 'badge-low' : 'badge-medium'}`} style={{ fontSize: '0.7rem' }}>
                          {m.is_active ? 'Active Engine' : 'Available'}
                        </span>
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--color-text)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <FileCode size={14} color="var(--color-primary)" />
                          <span>{m.filename}</span>
                        </div>
                      </td>
                      <td>
                        <span className="badge badge-medium" style={{ fontSize: '0.7rem' }}>
                          {m.format}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{m.size_mb} MB</td>
                      <td style={{ color: 'var(--color-text-secondary)', fontSize: '0.75rem' }}>{m.modified_at}</td>
                      <td>
                        {m.is_active ? (
                          <span style={{ fontSize: '0.75rem', color: '#16a34a', fontWeight: 600 }}>Active</span>
                        ) : (
                          <button
                            onClick={() => handleActivate(m.filename)}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                          >
                            <ArrowRight size={12} />
                            <span>Activate</span>
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: '24px', color: 'var(--color-text-muted)' }}>
                      Loading model weights registry...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 3. Cross-Domain Evaluation Benchmarks */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Award size={16} />
            <span>Cross-Domain Generalization Benchmarks (Unseen Test Domains)</span>
          </div>
          <span className="badge badge-low" style={{ fontSize: '0.7rem' }}>
            Zero-Leakage Test Splits
          </span>
        </div>

        <div className="card-body" style={{ padding: 0 }}>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Operating Domain</th>
                  <th>Test Dataset Split</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>mAP@50</th>
                  <th>mAP@50-95</th>
                  <th>Inference Latency</th>
                  <th>Optical Environmental Challenges</th>
                </tr>
              </thead>
              <tbody>
                {domainBenchmarks.map((bm, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 600, color: 'var(--color-text)' }}>{bm.domain}</td>
                    <td style={{ fontSize: '0.78rem' }}>{bm.location}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#16a34a' }}>{bm.precision}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#16a34a' }}>{bm.recall}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--color-primary)' }}>{bm.map50}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{bm.map5095}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>{bm.latency}</td>
                    <td style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', maxWidth: 280 }}>{bm.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 4. Two-Column Section: Edge Acceleration & Plastic Category Ontology */}
      <div className="grid-2">
        {/* Runtime Acceleration: PyTorch vs ONNX Runtime */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Zap size={16} />
              <span>Edge Hardware Acceleration (PyTorch vs ONNX Runtime)</span>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                onClick={() => setSelectedRuntime('pytorch')}
                className={`btn btn-sm ${selectedRuntime === 'pytorch' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 10px', fontSize: '0.72rem' }}
              >
                PyTorch Native
              </button>
              <button
                onClick={() => setSelectedRuntime('onnx')}
                className={`btn btn-sm ${selectedRuntime === 'onnx' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '3px 10px', fontSize: '0.72rem' }}
              >
                ONNX Runtime
              </button>
            </div>
          </div>

          <div className="card-body">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 6 }}>
                  <span>Inference Latency</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {selectedRuntime === 'onnx' ? '8.2 ms (121.9 FPS)' : '18.4 ms (54.3 FPS)'}
                  </span>
                </div>
                <div style={{ height: 6, borderRadius: 3, backgroundColor: 'var(--color-surface-muted)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: selectedRuntime === 'onnx' ? '92%' : '48%',
                      backgroundColor: selectedRuntime === 'onnx' ? '#16a34a' : 'var(--color-primary)',
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 6 }}>
                  <span>RAM / VRAM Footprint</span>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {selectedRuntime === 'onnx' ? '165 MB (Optimized)' : '420 MB (Torch Runtime)'}
                  </span>
                </div>
                <div style={{ height: 6, borderRadius: 3, backgroundColor: 'var(--color-surface-muted)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: selectedRuntime === 'onnx' ? '39%' : '85%',
                      backgroundColor: selectedRuntime === 'onnx' ? 'var(--color-primary)' : '#d97706',
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>

              <div style={{ padding: 10, backgroundColor: 'var(--color-surface-muted)', borderRadius: 'var(--radius-md)', fontSize: '0.75rem', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                {selectedRuntime === 'onnx'
                  ? 'ONNX Runtime is ideal for edge microcontrollers (Jetson Orin, Raspberry Pi 5) with graph constant folding.'
                  : 'PyTorch Native runtime provides dynamic batching, direct tensor graph interoperability, and training backpropagation.'}
              </div>
            </div>
          </div>
        </div>

        {/* Marine Plastic Ontology & Classes */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Layers size={16} />
              <span>Marine Plastic Category Ontology</span>
            </div>
            <span className="badge badge-low" style={{ fontSize: '0.7rem' }}>
              5 Target Classes
            </span>
          </div>

          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {modelClasses.map((mc) => (
              <div
                key={mc.name}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--color-surface-muted)',
                  border: '1px solid var(--color-border)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.78rem', color: 'var(--color-text)' }}>
                    {mc.name.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                    {mc.desc}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className={`badge ${mc.risk === 'Critical' ? 'badge-critical' : mc.risk === 'High' ? 'badge-high' : 'badge-medium'}`} style={{ fontSize: '0.65rem' }}>
                    {mc.risk}
                  </span>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: 2 }}>
                    Fill: {Math.round(mc.fillFactor * 100)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
