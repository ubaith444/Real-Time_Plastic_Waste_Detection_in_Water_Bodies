"""
System Launcher: AI-Based Real-Time Plastic Waste Detection System
Starts the FastAPI application and background computer vision pipeline.
"""

import sys
import os
import subprocess
import uvicorn

def main():
    print("==================================================================")
    print("AI-BASED REAL-TIME PLASTIC WASTE DETECTION IN WATER BODIES")
    print("Computer Vision Monitoring: Webcam, Drone, Boat, and Underwater")
    print("==================================================================")
    print("Starting FastAPI Backend & Inference Engine on http://localhost:8000 ...")
    print("Monitoring Dashboard available at: http://localhost:8000")
    print("REST API Docs available at:        http://localhost:8000/docs")
    print("==================================================================")

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
