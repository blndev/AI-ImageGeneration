# app/ui/state/session_manager.py
from datetime import datetime, timedelta
import logging
from app import SessionState

from app.utils.singleton import singleton

from app.analytics import Analytics


# Set up module logger
logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_references = {}
        self.last_generation = datetime.now()

    def record_session_as_active(self, session_state: SessionState):
        if self.active_sessions is not None:
            self.active_sessions[session_state.session] = datetime.now()

    def cleanup_inactive_sessions(self, timeout_minutes):
        x15_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)
        to_be_removed = []
        for key, last_active in self.active_sessions.items():
            if last_active < x15_minutes_ago:
                to_be_removed.append(key)
        return to_be_removed
