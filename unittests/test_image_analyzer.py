import unittest
import random, uuid, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import OllamaImageAnalyzer

class TestGradioUISessionState(unittest.TestCase):
    def setUp(self):
        self.image_analyzer = OllamaImageAnalyzer()

    def test_simple(self):
        """Test SessionState."""
        answer = self.image_analyzer.analyze_image("TBD")
        self.assertIsNotNone(answer, "there must be  always an answer")
        
        self.assertIsNotNone(answer.get("human"), "there should be a human in this image")
