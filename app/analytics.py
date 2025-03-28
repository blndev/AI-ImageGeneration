from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging
from typing import Optional, Tuple

from user_agents import parse as parse_user_agent   # Split OS. Browser etc.
from app.utils.singleton import singleton

logger = logging.getLogger(__name__)

@singleton
class Analytics:
    """
    Analytics class for tracking and monitoring application metrics using Prometheus.
    Implements singleton pattern to ensure only one instance exists.
    """
    
    def __init__(self):
        """Initialize Analytics with Prometheus metrics and start the HTTP server."""
        try:
            logger.info("Initializing Analytics")
            # Initialize Prometheus metrics
            self._image_creations = Counter(
                'flux_image_creations_total',
                'Total number of images created',
                labelnames=('model', 'content', 'reference_code')
            )

            self._sessions = Counter(
                'flux_sessions_total',
                'Total number of user sessions',
                labelnames=('device_type', 'os', 'browser', 'language', 'reference_code')
            )

            self._active_sessions = Gauge(
                'flux_active_sessions',
                'Amount of users active in the last 30 minutes'
            )

            self._image_creation_time = Histogram(
                'flux_image_creation_duration_seconds',
                'Time taken to create an image',
                buckets=[1, 2, 5, 10, 20, 30, 60]  # buckets in seconds
            )

            self._user_tokens = Gauge(
                'flux_user_tokens',
                'Current number of tokens available to users',
                ['user_id']
            )

            # Start Prometheus HTTP server on port 9101
            start_http_server(9101)
        except Exception as e:
            logger.error(f"Failed to initialize Analytics: {e}")
            raise

    def record_image_creation(self, count: int = 1, model: str = "unknown", 
                            content: str = "unknown", reference: str = "") -> None:
        """
        Record a new image creation event.

        Args:
            count (int): Number of images created (default: 1)
            model (str): Model used for image creation (default: "unknown")
            content (str): Content type or description (default: "unknown")
            reference (str): Reference code for the creation (default: "")
        """
        try:
            self._image_creations.labels(
                model=model,
                content=content,
                reference_code=reference
            ).inc(amount=count)
        except Exception as e:
            logger.warning(f"Failed to record image creation: {e}")

    def parse_user_agent(self, user_agent: str = "", 
                        languages: str = "") -> Tuple[str, str, str, str]:
        """
        Parse user agent string to extract OS, browser, device type, and language.

        Args:
            user_agent (str): User agent string from the request (default: "")
            languages (str): Language string from the request (default: "")

        Returns:
            Tuple[str, str, str, str]: Tuple containing (os, browser, device_type, language)
        """
        os = 'Unknown'
        browser = 'Unknown'
        device_type = 'Unknown'
        language = 'Unknown'

        try:
            if user_agent:
                ua = parse_user_agent(user_agent)
                os = ua.os.family
                browser = ua.browser.family
                device_type = ua.device.family
                if device_type == 'Other':
                    device_type = 'Desktop'
                if 'Mobile' in ua.device.family:
                    device_type = 'Mobile'

            if languages:
                language = languages.split(',')[0]

        except Exception as e:
            logger.warning(f"Failed to parse user agent: {e}")

        return os, browser, device_type, language

    def record_new_session(self, user_agent: str = "", 
                          languages: str = "", 
                          reference: str = "") -> None:
        """
        Record a new user session.

        Args:
            user_agent (str): User agent string from the request (default: "")
            languages (str): Language string from the request (default: "")
            reference (str): Reference code for the session (default: "")
        """
        try:
            os, browser, dt, lng = self.parse_user_agent(user_agent, languages)
            self._sessions.labels(
                os=os,
                browser=browser,
                device_type=dt,
                language=lng,
                reference_code=reference
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record new session: {e}")

    def update_active_sessions(self, sessioncount: int) -> None:
        """
        Update the count of active sessions.

        Args:
            sessioncount (int): Current number of active sessions
        """
        try:
            self._active_sessions.set(sessioncount)
        except Exception as e:
            logger.warning(f"Failed to update active sessions: {e}")

    def start_image_creation_timer(self) -> Optional[float]:
        """
        Start timing an image creation.

        Returns:
            float: Current timestamp or None if operation fails
        """
        try:
            return time.time()
        except Exception as e:
            logger.warning(f"Failed to start image creation timer: {e}")
            return None

    def stop_image_creation_timer(self, start_time: Optional[float]) -> None:
        """
        Stop timing an image creation and record the duration.

        Args:
            start_time (float, optional): Start time returned by start_image_creation_timer
        """
        if start_time is None:
            return
        
        try:
            duration = time.time() - start_time
            self._image_creation_time.observe(duration)
        except Exception as e:
            logger.warning(f"Failed to stop image creation timer: {e}")

    def update_user_tokens(self, user_id: str, tokens: int) -> None:
        """
        Update the token count for a user.

        Args:
            user_id (str): Unique identifier for the user
            tokens (int): Number of tokens available to the user
        """
        try:
            self._user_tokens.labels(user_id=user_id).set(tokens)
        except Exception as e:
            logger.warning(f"Failed to update user tokens: {e}")
