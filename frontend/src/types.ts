export interface Camera {
  id: string;
  name: string;
  environment_type: 'webcam' | 'drone' | 'boat' | 'underwater';
  source_url: string;
  location_name: string;
  latitude: number;
  longitude: number;
  is_active: boolean;
  fps: number;
  resolution: string;
  created_at?: string;
}

export interface DetectionEvent {
  id: number;
  camera_id: string;
  track_id: number;
  class_name: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  snapshot_path: string | null;
  environmental_setting: string | null;
  dwell_time_sec: number;
  timestamp: string;
}

export interface Alert {
  id: number;
  camera_id: string;
  track_id: number;
  severity: 'Critical' | 'High' | 'Medium';
  class_name: string;
  message: string;
  snapshot_path: string | null;
  acknowledged: boolean;
  timestamp: string;
}

export interface AnalyticsSummary {
  total_detections: number;
  unique_objects_tracked: number;
  total_alerts: number;
  active_cameras: number;
  class_breakdown: Record<string, number>;
  severity_breakdown: Record<string, number>;
  camera_breakdown: Record<string, number>;
  recent_detections_timeline: Array<{
    id: number;
    track_id: number;
    class_name: string;
    camera_id: string;
    dwell_time_sec: number;
    timestamp: string;
  }>;
}

export interface MapMarker {
  camera_id: string;
  name: string;
  environment_type: string;
  location_name: string;
  latitude: number;
  longitude: number;
  is_active: boolean;
  total_detections: number;
  critical_alerts: number;
  pollution_level: 'Low' | 'Moderate' | 'High' | 'Critical';
}

export interface StreamTelemetry {
  camera_id: string;
  camera_name: string;
  environment_type: string;
  gps: [number, number];
  fps: number;
  latency_ms: number;
  debris_count: number;
  sahi_enabled?: boolean;
  sahi_mode?: string;
  sahi_slices?: number;
  tracker_mode?: string;
  water_current_vector?: [number, number];
  coverage_percentage?: number;
  area_sq_meters?: number;
  density_status?: string;
  density_badge?: string;
  drone_altitude_m?: number | null;
  dynamic_gsd_cm_per_px?: number;
  slick_area_m2?: number;
  containment_boom_meters?: number;
  hull_points?: Array<[number, number]>;
  tracks: Array<{
    track_id: number;
    class_name: string;
    confidence: number;
    dwell_time_sec: number;
    is_occluded?: boolean;
    velocity?: [number, number];
    box: [number, number, number, number];
    normalized_box: [number, number, number, number];
  }>;
  recent_alert_count: number;
  timestamp: number;
}

export interface UsbDevice {
  device_index: number;
  name: string;
  resolution: string;
  fps: number;
  is_available: boolean;
}
