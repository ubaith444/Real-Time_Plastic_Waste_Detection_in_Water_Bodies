"""
Alert Notification Dispatcher
Handles configurable incident notifications, webhook payloads, and simulated email dispatches.
"""

import time

class NotificationService:
    def __init__(self):
        self.notification_history: list[dict] = []
        self.webhook_url: str | None = None
        self.email_alerts_enabled: bool = True
        self.notification_recipients: list[str] = ["marine-incident-team@cleanup.org"]

    def dispatch_alert(self, alert_data: dict):
        entry = {
            "id": len(self.notification_history) + 1,
            "timestamp": time.time(),
            "severity": alert_data.get("severity", "Medium"),
            "camera_id": alert_data.get("camera_id"),
            "message": alert_data.get("message"),
            "dispatched_to": self.notification_recipients if self.email_alerts_enabled else [],
            "status": "delivered"
        }
        self.notification_history.append(entry)
        if len(self.notification_history) > 200:
            self.notification_history.pop(0)

    def get_recent_dispatches(self) -> list[dict]:
        return list(reversed(self.notification_history[-50:]))

notifier = NotificationService()
