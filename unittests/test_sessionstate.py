import unittest
from datetime import datetime, timedelta
import json
import uuid
from app.SessionState import SessionState


class TestSessionState(unittest.TestCase):
    """Test cases for SessionState class"""

    def test_init_default(self):
        """Test initialization with default values"""
        state = SessionState()
        self.assertEqual(state.token, 0)
        self.assertIsNotNone(state.session)
        self.assertIsNone(state.last_generation)
        self.assertEqual(state.nsfw, 0)
        self.assertIsNone(state.reference_code)

    def test_init_custom(self):
        """Test initialization with custom values"""
        session_id = str(uuid.uuid4())
        last_gen = datetime.now().isoformat()
        ref_code = "test-ref-123"
        
        state = SessionState(
            token=10,
            session=session_id,
            last_generation=last_gen,
            nsfw=1,
            reference_code=ref_code
        )
        
        self.assertEqual(state.token, 10)
        self.assertEqual(state.session, session_id)
        self.assertEqual(state.last_generation, last_gen)
        self.assertEqual(state.nsfw, 1)
        self.assertEqual(state.reference_code, ref_code)

    def test_session_property(self):
        """Test session property getter and setter"""
        state = SessionState()
        new_session = str(uuid.uuid4())
        state.session = new_session
        self.assertEqual(state.session, new_session)
        
        # Test type validation
        with self.assertRaises(TypeError):
            state.session = 12345

    def test_to_dict(self):
        """Test conversion to dictionary"""
        state = SessionState(
            token=5,
            session="test-session",
            last_generation="2025-01-01T12:00:00",
            nsfw=1,
            reference_code="ref-123"
        )
        
        result = state.to_dict()
        
        self.assertEqual(result["token"], 5)
        self.assertEqual(result["session"], "test-session")
        self.assertEqual(result["last_generation"], "2025-01-01T12:00:00")
        self.assertEqual(result["nsfw"], 1)
        self.assertEqual(result["reference_code"], "ref-123")

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "token": 5,
            "session": "test-session",
            "last_generation": "2025-01-01T12:00:00",
            "nsfw": 1,
            "reference_code": "ref-123"
        }
        
        state = SessionState.from_dict(data)
        
        self.assertEqual(state.token, 5)
        self.assertEqual(state.session, "test-session")
        self.assertEqual(state.last_generation, "2025-01-01T12:00:00")
        self.assertEqual(state.nsfw, 1)
        self.assertEqual(state.reference_code, "ref-123")
        
        # Test with empty data
        state = SessionState.from_dict(None)
        self.assertEqual(state.token, 0)
        self.assertIsNotNone(state.session)

    def test_from_gradio_state(self):
        """Test creation from gradio state string"""
        data = {
            "token": 5,
            "session": "test-session",
            "last_generation": "2025-01-01T12:00:00",
            "nsfw": 1,
            "reference_code": "ref-123"
        }
        
        json_str = json.dumps(data)
        state = SessionState.from_gradio_state(json_str)
        
        self.assertEqual(state.token, 5)
        self.assertEqual(state.session, "test-session")
        self.assertEqual(state.last_generation, "2025-01-01T12:00:00")
        self.assertEqual(state.nsfw, 1)
        self.assertEqual(state.reference_code, "ref-123")
        
        # Test with empty data
        state = SessionState.from_gradio_state(None)
        self.assertEqual(state.token, 0)
        self.assertIsNotNone(state.session)
        
        # Test with invalid data
        with self.assertRaises(Exception):
            SessionState.from_gradio_state("invalid json")

    def test_str_and_repr(self):
        """Test string representation methods"""
        state = SessionState(token=5, session="test-session")
        str_repr = str(state)
        repr_str = repr(state)
        
        self.assertEqual(str_repr, repr_str)
        self.assertIn("test-session", str_repr)
        self.assertIn("5", str_repr)

    def test_save_and_reset_generation_activity(self):
        """Test saving and resetting generation activity"""
        state = SessionState()
        
        # Initial state
        self.assertIsNone(state.last_generation)
        
        # Save activity
        state.save_last_generation_activity()
        self.assertIsNotNone(state.last_generation)
        
        # Reset activity
        state.reset_last_generation_activity()
        self.assertIsNone(state.last_generation)

    def test_generation_before_minutes(self):
        """Test generation_before_minutes method"""
        state = SessionState()
        
        # Test with None
        self.assertFalse(state.generation_before_minutes(5))
        
        # Test with empty string
        state.last_generation = ""
        self.assertFalse(state.generation_before_minutes(5))
        
        # Test with "None" string
        state.last_generation = "None"
        self.assertFalse(state.generation_before_minutes(5))
        
        # Test with recent timestamp (less than 5 minutes ago)
        state.last_generation = datetime.now().isoformat()
        self.assertFalse(state.generation_before_minutes(5))
        
        # Test with old timestamp (more than 5 minutes ago)
        old_time = datetime.now() - timedelta(minutes=10)
        state.last_generation = old_time.isoformat()
        self.assertTrue(state.generation_before_minutes(5))
        
        # Test with invalid timestamp
        state.last_generation = "invalid-timestamp"
        self.assertTrue(state.generation_before_minutes(5))

    def test_reference_code(self):
        """Test reference code methods"""
        # Test without reference code
        state = SessionState()
        self.assertFalse(state.has_reference_code())
        
        # Test with empty reference code
        state.reference_code = ""
        self.assertFalse(state.has_reference_code())
        
        # Test with valid reference code
        state.reference_code = "test-ref"
        self.assertTrue(state.has_reference_code())
        self.assertEqual(state.get_reference_code(), "test-ref")
        
        # Test get_reference_code auto-generation
        state.reference_code = None
        ref_code = state.get_reference_code()
        self.assertIsNotNone(ref_code)
        self.assertTrue(state.has_reference_code())


if __name__ == "__main__":
    unittest.main()