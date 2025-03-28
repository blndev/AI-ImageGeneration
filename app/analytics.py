from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

from user_agents import parse as parse_user_agent   # Split OS. Browser etc.
from app.utils.singleton import singleton

logger = logging.getLogger(__name__)


@singleton
class Analytics():
    def __init__(self):
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

    def record_image_creation(self, count: int = 1, model: str = None, content: str = None, reference: str = None):
        """Record a new image creation"""
        if not reference: reference = ""
        self._image_creations.labels(model=model, content=content, reference_code=reference).inc(amount=count)

    def parse_user_agent(self, user_agent, languages):
        """Parse user agent string to extract OS, browser, device type, and language"""
        os = 'Unknown'
        browser = 'Unknown'
        device_type = 'Unknown'
        language = 'Unknown'

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

        return os, browser, device_type, language

    def record_new_session(self, user_agent: str = None, languages: str = None, reference: str = None):
        """Record a new user session"""
        try:
            os, browser, dt, lng = self.parse_user_agent(user_agent, languages)
            if not reference: reference = ""
            self._sessions.labels(os=os, browser=browser, device_type=dt, language=lng, reference_code=reference).inc()
        except Exception as e:
            logger.warning(f"Error while recording new session: {e}")

    def update_active_sessions(self, sessioncount: int):
        """Update the token count for a user"""
        self._active_sessions.set(sessioncount)

    def start_image_creation_timer(self) -> time.time:
        """Start timing an image creation"""
        return time.time()

    def stop_image_creation_timer(self, start_time):
        """Stop timing an image creation and record the duration"""
        if start_time is None: return
        try:
            duration = time.time() - start_time
            self._image_creation_time.observe(duration)
        except Exception as e:
            logger.warning(f"Error while stopping image creation timer and posting result: {e}")

    def update_user_tokens(self, user_id: str, tokens: int):
        """Update the token count for a user"""
        self._user_tokens.labels(user_id=user_id).set(tokens)

# # Initialize analytics when module is imported
# analytics = Analytics()
