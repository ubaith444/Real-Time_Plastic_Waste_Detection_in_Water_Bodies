from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.cv.camera_stream import stream_manager

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Real-Time Plastic Waste Detection System" in data["service"]
    assert data["sahi_batched"] is True

def test_frontend_mount():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_cameras_endpoint():
    response = client.get("/api/cameras")
    assert response.status_code == 200
    cameras = response.json()
    assert len(cameras) >= 4
    cam_ids = [c["id"] for c in cameras]
    assert "drone-01" in cam_ids
    assert "boat-01" in cam_ids
    assert "underwater-01" in cam_ids
    assert "webcam-01" in cam_ids

def test_analytics_summary_endpoint():
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_detections" in data
    assert "class_breakdown" in data
    assert "severity_breakdown" in data

def test_alerts_endpoint():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_quad_and_station_frame_generation():
    # Test that quad grid frame is generated with valid JPEG bytes
    quad_bytes = stream_manager.get_quad_frame_jpeg()
    assert len(quad_bytes) > 1000
    assert quad_bytes[:2] == b'\xff\xd8'  # Valid JPEG magic bytes

    # Test that station frame is generated with valid JPEG bytes
    for env in ["drone", "boat", "underwater", "webcam"]:
        station_bytes = stream_manager.get_station_frame_jpeg(env)
        assert len(station_bytes) > 1000
        assert station_bytes[:2] == b'\xff\xd8'

if __name__ == "__main__":
    test_health_check()
    test_frontend_mount()
    test_cameras_endpoint()
    test_analytics_summary_endpoint()
    test_alerts_endpoint()
    test_quad_and_station_frame_generation()
    print("test_api passed successfully.")
