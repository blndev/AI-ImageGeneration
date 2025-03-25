from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time, logging

from user_agents import parse as parse_user_agent   # Split OS. Browser etc.
from app.utils.singleton import singleton

logger = logging.getLogger(__name__)

@singleton
class Analytics():
    def __init__(self):
        logger.info("Initializing Analytics")
        # Initialize Prometheus metrics
        self.image_creations = Counter(
            'flux_image_creations_total',
            'Total number of images created',
            labelnames=('model', 'content')
        )
        
        self.sessions = Counter(
            'flux_sessions_total',
            'Total number of user sessions',
            labelnames=('device_type', 'os', 'browser', 'language')
        )
        
        self.image_creations_by_reference = Counter(
            'flux_image_creations_total_reference',
            'Total number of images created via reference', 
            labelnames=('reference_code',)
        )

        self.active_sessions = Gauge(
            'flux_active_sessions_total',
            'Amount of users active in the last 30 minutes'
        )

        self.image_creation_time = Histogram(
            'flux_image_creation_duration_seconds',
            'Time taken to create an image',
            buckets=[1, 2, 5, 10, 20, 30, 60]  # buckets in seconds
        )
        
        self.user_tokens = Gauge(
            'flux_user_tokens',
            'Current number of tokens available to users',
            ['user_id']
        )
        
        # Start Prometheus HTTP server on port 9101
        start_http_server(9101)
    
    def record_image_creation(self, model:str=None, content: str = None):
        """Record a new image creation"""
        self.image_creations.labels(model=model, country=content).inc()
    
    def record_image_creation_by_reference(self, reference:str=None):
        """Record a new image creation"""
        self.image_creations_by_reference.labels(reference_code=reference).inc()
    
    def parse_user_agent(self, user_agent, languages):
        """Parse user agent string to extract OS, browser, device type, and language"""
        os = 'Unknown'
        browser = 'Unknown'
        device_type = 'Unknown'
        language = 'Unknown'

        if user_agent:
            ua = parse_user_agent(user_agent)
            os =  ua.os.family
            browser = ua.browser.family
            device_type = ua.device.family
            if device_type == 'Other':
                device_type = 'Desktop'
            if 'Mobile' in ua.device.family:
                device_type = 'Mobile'

        if languages:
            language = languages.split(',')[0]

        return os, browser, device_type, language

    def record_new_session(self, user_agent: str = None, languages: str=None):
        """Record a new user session"""
        try:
            os, browser, dt, lng = self.parse_user_agent(user_agent, languages)
            self.sessions.labels(os=os, browser=browser, device_type=dt, language=lng).inc()
            #logger.debug(f"New user session recorded. Now: {self.sessions._value.get()}")
        except Exception as e:
            logger.warning(f"Error while recording new session: {e}")
    
    def start_image_creation_timer(self):
        """Start timing an image creation"""
        return time.time()
    
    def stop_image_creation_timer(self, start_time):
        """Stop timing an image creation and record the duration"""
        try:
            duration = time.time() - start_time
            self.image_creation_time.observe(duration)
        except Exception as e:
            logger.warning(f"Error while stopping image creation timer and posting result: {e}")
    
    def update_user_tokens(self, user_id: str, tokens: int):
        """Update the token count for a user"""
        self.user_tokens.labels(user_id=user_id).set(tokens)

# # Initialize analytics when module is imported
# analytics = Analytics()
