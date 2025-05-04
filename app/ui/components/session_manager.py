# app/ui/state/session_manager.py
from datetime import datetime, timedelta
import logging
from app import SessionState

from app.utils.singleton import singleton
from app.appconfig import AppConfig
from app.analytics import Analytics


# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class SessionManager:
    """
    contains all active sessions for analytics as well as
    all code to handle the time based token generation if available
    """

    def __init__(self, config: AppConfig, analytics: Analytics):
        self.config = config
        self.analytics = analytics

        self.active_sessions = {}  # key=sessionid, value = timestamp of last token refresh
        # TODO move to link_handler
        self.session_references = {}  # key=referencID, value = int count(image created via reference)
        logger.debug("Initial token: %i, wait time: %i minutes", self.config.initial_token, self.config.new_token_wait_time)

    def session_cleanup_and_analytics(self):
        """is called every 60 secdonds and:
        * updates monitoring information
        * refreshes configuration
        * unloading unused models
        """
        logger.debug("session_cleanup_and_analytics")
        timeout_minutes = 15  # self.config.free_memory_after_minutes_inactivity
        timestamp_x_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)

        to_be_removed = []
        for key, last_active in self.active_sessions.items():
            if last_active < timestamp_x_minutes_ago:
                to_be_removed.append(key)

        if len(to_be_removed) > 0:
            logger.info(f"remove {len(to_be_removed)} sessions as they are inactive for {timeout_minutes} minutes")

        for ktr in to_be_removed:
            self.active_sessions.pop(ktr)

        # report stats
        self.analytics.update_active_sessions(len(self.active_sessions))

    def record_session_as_active(self, session_state: SessionState):
        if self.active_sessions is not None:
            self.active_sessions[session_state.session] = datetime.now()
        self.analytics.update_active_sessions(len(self.active_sessions))

    def check_new_token_after_wait_time(self, session_state: SessionState):
        logger.debug(f"check new token for '{session_state.session}'. Last Generation: {session_state.last_generation}")
        # logic: (10) minutes after running out of token, user get filled up to initial (10) new token
        # exception: user upload image for training or receive advertising token
        self.record_session_as_active(session_state)
        if self.config.token_enabled:
            current_token = session_state.token
            if session_state.generation_before_minutes(self.config.new_token_wait_time) and session_state.token <= 2:
                session_state.token = self.config.initial_token
                session_state.reset_last_generation_activity()
            new_token = session_state.token - current_token
            if new_token > 0:
                logger.info(f"session {session_state.session} received {new_token} new token for waiting")
            return session_state, new_token

    # def cleanup_inactive_sessions(self, timeout_minutes):
    #     x15_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)
    #     to_be_removed = []
    #     for key, last_active in self.active_sessions.items():
    #         if last_active < x15_minutes_ago:
    #             to_be_removed.append(key)
    #     return to_be_removed
