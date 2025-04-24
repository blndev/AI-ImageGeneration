# app/ui/state/session_manager.py
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_references = {}
        self.last_generation = datetime.now()

    def record_session_as_active(self, session_state):
        if self.active_sessions is not None:
            self.active_sessions[session_state.session] = datetime.now()

    def cleanup_inactive_sessions(self, timeout_minutes):
        x15_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)
        to_be_removed = []
        for key, last_active in self.active_sessions.items():
            if last_active < x15_minutes_ago:
                to_be_removed.append(key)
        return to_be_removed
