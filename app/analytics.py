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
                'imggen_image_creations',
                'number of images created',
                labelnames=('model', 'content')
            )
            self._image_creations.labels('', '')

            self._nsfw_image_creations = Counter(
                'imggen_image_nsfw_creations',
                'number of nsfw images detected',
                labelnames=('model', 'content')
            )

            self._prompt_freestyle = Counter(
                'imggen_prompt_freestyle_used',
                'number of freestyle prompt usage',
                ['magicprompt']
            )

            self._reference_usage = Counter(
                'imggen_image_creation_with_reference',
                'number of images created by a reference link',
                ['reference']
            )

            self._prompt_assistant = Counter(
                'imggen_prompt_assistant_used',
                'number of assistant based prompt usage',
                ['magicprompt']
            )

            self._sessions = Counter(
                'imggen_sessions_total',
                'Total number of user sessions',
                labelnames=('device_type', 'os', 'browser', 'language', 'reference_code')
            )

            self._uploads = Counter(
                'imggen_uploads_total',
                'Total number of uploaded files',
                labelnames=('device_type', 'os', 'browser', 'language', 'content')
            )

            self._errors = Counter(
                'imggen_errors',
                'Total number of occured errors',
                labelnames=('module', 'criticality')
            )

            self._active_sessions = Gauge(
                'imggen_active_sessions',
                'Amount of users active in the last 30 minutes'
            )

            self._image_creation_time = Histogram(
                'imggen_image_creation_duration_seconds',
                'Time taken to create an image including queue wait time',
                buckets=[1, 2, 5, 10, 20, 30, 60]  # buckets in seconds
            )

            self._user_tokens = Gauge(
                'imggen_user_tokens',
                'Current number of tokens available to users',
                ['user_id']
            )

            # Start Prometheus HTTP server on port 9101
            start_http_server(9101)
        except Exception as e:
            logger.error(f"Failed to initialize Analytics: {e}")

    def record_application_error(self, module: str, criticality: str):
        """
        criticality = warning, error, critical
        """
        try:
            self._errors.labels(
                module=module,
                criticality=criticality.lower()
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record application error: {e}")

    def record_reference_usage(self, shared_reference_key, image_count):
        try:
            self._reference_usage.labels(
                reference=shared_reference_key,
            ).inc(amount=image_count)
        except Exception as e:
            logger.error(f"Failed to record reference_usage: {e}")

    def record_image_creation(self,
                              count: int = 1,
                              nsfw_count: int = 0,
                              model: str = "unknown",
                              content: str = "",
                              ) -> None:
        """
        Record a new image creation event.

        Args:
            count (int): Number of images created (default: 1)
            nsfw_count (int): Number of images detected as nsfw  (default: 0)
            model (str): Model used for image creation (default: "unknown")
            content (str): Content type or description (default: "unknown")
        """
        try:
            self._image_creations.labels(
                model=model,
                content=content
            ).inc(amount=count)
            if nsfw_count > 0:
                self._nsfw_image_creations.labels(
                    model=model,
                    content=content
                ).inc(amount=nsfw_count)
        except Exception as e:
            logger.warning(f"Failed to record image creation: {e}")

    def _parse_user_agent(self,
                          user_agent: str = "",
                          languages: str = ""
                          ) -> Tuple[str, str, str, str]:
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

    def record_new_session(self,
                           user_agent: str = "",
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
            if reference is None: reference = ""
            reference = reference.split()
            os, browser, dt, lng = self._parse_user_agent(user_agent, languages)
            self._sessions.labels(
                os=os,
                browser=browser,
                device_type=dt,
                language=lng,
                reference_code=reference
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record new session: {e}")

    def record_new_upload(self,
                          user_agent: str = "",
                          languages: str = "",
                          content: str = "") -> None:
        """
        Record a new user upload.

        Args:
            user_agent (str): User agent string from the request (default: "")
            languages (str): Language string from the request (default: "")
            content (str): Type of content e.g. "ai", "to_small", "sfw", "teasing", explicit based on content detection
        """
        try:
            os, browser, dt, lng = self._parse_user_agent(user_agent, languages)
            self._uploads.labels(
                os=os,
                browser=browser,
                device_type=dt,
                language=lng,
                content=content
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record new session: {e}")

    def record_prompt_usage(self, promptmagic_used: bool, assistant_used: bool, image_count: int = 1) -> None:
        """
        Record a new prompt usage (generation request).

        Args:
            promptmagic_used (bool): true if prompt magic was active
            assistant_used (bool): true if teh assistant was used, false = freestyle prompt
        """
        try:
            logger.debug(f"record prompt usage Assistant: {assistant_used}, Prompt Magic: {promptmagic_used}")
            if assistant_used:
                # print("assistant")
                # p = self._prompt_assistant
                # print (p._labelnames)
                # print(self._uploads._labelnames)
                # p.labels(magicprompt="True").inc()
                self._prompt_assistant.labels(
                    magicprompt=promptmagic_used
                ).inc(amount=image_count)
            else:
                # print("frestyle")
                self._prompt_freestyle.labels(
                    magicprompt=promptmagic_used
                ).inc(amount=image_count)

        except Exception as e:
            logger.warning(f"Failed to record a prompt usage: {e}")
            raise e

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

    def stop_image_creation_timer(self, start_time: Optional[float], image_count: int = 1) -> None:
        """
        Stop timing an image creation and record the duration.

        Args:
            start_time (float, optional): Start time returned by start_image_creation_timer
        """
        if start_time is None:
            return

        try:
            duration = (time.time() - start_time) / image_count
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
