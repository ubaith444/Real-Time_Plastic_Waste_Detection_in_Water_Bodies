"""
Telemetry and Incident Alert Pub/Sub Event Broker
Provides multi-subscriber broadcasting across active dashboard operator sessions.
Implements an in-memory queue bus with optional Redis pub/sub adapter readiness.
"""

import asyncio
import json
import threading
from typing import Callable, Any

class TelemetryPubSub:
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._latest_telemetry: dict = {}

    def publish_telemetry(self, data: dict):
        """
        Broadcasts telemetry payload to all active session queues.
        """
        with self._lock:
            self._latest_telemetry = data
            dead_queues = []
            for q in list(self._subscribers):
                try:
                    # Non-blocking put; if consumer is slow, drop stale frame
                    if q.full():
                        try:
                            _ = q.get_nowait()
                        except Exception:
                            pass
                    q.put_nowait(data)
                except Exception:
                    dead_queues.append(q)

            for dead in dead_queues:
                self._subscribers.discard(dead)

    def subscribe(self, max_buffer: int = 5) -> asyncio.Queue:
        """
        Registers an async subscriber queue for real-time telemetry streaming.
        """
        q = asyncio.Queue(maxsize=max_buffer)
        with self._lock:
            self._subscribers.add(q)
            if self._latest_telemetry:
                try:
                    q.put_nowait(self._latest_telemetry)
                except Exception:
                    pass
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """
        Removes subscriber queue on disconnect.
        """
        with self._lock:
            self._subscribers.discard(q)

    def active_subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

telemetry_bus = TelemetryPubSub()
