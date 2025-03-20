from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from app.utils.singleton import Singleton

class Analytics(metaclass=Singleton):
    def __init__(self):
        # Initialize Prometheus metrics
        self.image_creations = Counter(
            'flux_image_creations_total',
            'Total number of images created'
        )
        
        self.sessions = Counter(
            'flux_sessions_total',
            'Total number of user sessions'
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
    
    def record_image_creation(self):
        """Record a new image creation"""
        self.image_creations.inc()
    
    def record_new_session(self):
        """Record a new user session"""
        self.sessions.inc()
    
    def start_image_creation_timer(self):
        """Start timing an image creation"""
        return time.time()
    
    def stop_image_creation_timer(self, start_time):
        """Stop timing an image creation and record the duration"""
        duration = time.time() - start_time
        self.image_creation_time.observe(duration)
    
    def update_user_tokens(self, user_id: str, tokens: float):
        """Update the token count for a user"""
        self.user_tokens.labels(user_id=user_id).set(tokens)

# Initialize analytics when module is imported
analytics = Analytics()
