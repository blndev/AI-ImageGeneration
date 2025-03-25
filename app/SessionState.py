from datetime import datetime
import json
import logging
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


class SessionState:
    """Object to handle session state information for the user"""

    @property
    def session(self) -> str:
        """Getter for 'session'-Attribut."""
        return self._session

    @session.setter
    def session(self, value):
        """Setter for 'session'-Attribut, makes sures it is always a string for serialization."""
        self._session = self._ensure_str(value)

    def _ensure_str(self, value) -> str:
        """Validates that types assigned to session are always strings."""
        if not isinstance(value, str):
            raise TypeError(f"session must be of type str, but {type(value).__name__} was assigned.")
        return value

    def __init__(self, token: int = 0, session: str = None, last_generation: str = None, reference_code: str =None):
        """State object for storing application data in browser."""
        self.token: int = token
        self.session: str = str(uuid.uuid4()) if session == None else session
        self.last_generation = last_generation
        self.reference_code = reference_code

    def __str__(self) -> str:
        """String representation for logging."""
        # gradio is using the (str) function to save the state instead of the repr
        return repr(self)  # f"SessionState(token={self.token}, session={self.session})"

    def __repr__(self) -> str:
        """Return a string representation of the SessionState."""
        return json.dumps(self.to_dict())

    def to_gradio_state(self) -> str:
        """Create SessionState from dictionary after deserialization."""
        return repr(self)

    def to_dict(self) -> dict:
        """Convert SessionState to dictionary for serialization."""
        return {
            'token': self.token,
            'session': str(self.session),
            'last_generation': self.last_generation,
            'reference_code': self.reference_code
        }

    def save_last_generation_activity(self):
        """Update last generation timestamp."""
        self.last_generation = datetime.now().isoformat()

    def reset_last_generation_activity(self):
        """Update last generation timestamp."""
        self.last_generation = None

    def generation_before_minutes(self, minutes: int) -> bool:
        """Check if last generation was before N minutes."""
        if self.last_generation is None or self.last_generation == "" or self.last_generation == "None":
            return False
        try:
            last_generation = datetime.fromisoformat(self.last_generation)
            return (datetime.now() - last_generation).total_seconds() > minutes * 60
        except Exception as e:
            logger.error(f"Error parsing last generation timestamp: {e}")
            return True

    def has_reference_code(self) -> bool:
        """Check if reference code is set."""
        return self.reference_code is not None and self.reference_code != ""
    
    def get_reference_code(self) -> str:
        """Get reference code."""
        if self.reference_code is None:
            self.reference_code = str(uuid.uuid4())
        return self.reference_code
    
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> 'SessionState':
        """Create SessionState from dictionary after deserialization."""
        if not data:
            return cls()
        return cls(
            token=data.get('token', 0),
            session=data.get('session', str(uuid.uuid4())),
            last_generation=data.get('last_generation', None),
            reference_code=data.get('reference_code', None)
        )

    @classmethod
    def from_gradio_state(cls, data: str) -> 'SessionState':
        """Create SessionState from dictionary string after deserialization."""
        if not data:
            return cls()
        if not isinstance(data, str):
            data = repr(data)
        try:
            d = json.loads(data)
            return cls.from_dict(d)
        except Exception:
            raise Exception(f"SessionState can't be converted from given value {data}")
