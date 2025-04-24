import unittest
from PIL import Image
import numpy as np
import os
import tempfile
from pathlib import Path
from app.validators import NSFWDetector, NSFWCategory, NSFWDetectionResult

class TestNSFWDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.detector = NSFWDetector(confidence_threshold=0.5)
        cls.test_dir = Path(tempfile.mkdtemp())
        cls._create_test_images()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after running tests"""
        # Remove test directory and all contents
        for file in cls.test_dir.glob("*"):
            file.unlink()
        cls.test_dir.rmdir()

    @classmethod
    def _create_test_images(cls):
        """Create test images with different characteristics"""
        # Create a simple RGB image
        cls.test_images = {}
        
        # Create blank image
        blank_img = Image.new('RGB', (224, 224), color='white')
        blank_path = cls.test_dir / "blank.jpg"
        blank_img.save(blank_path)
        cls.test_images['blank'] = str(blank_path)
        
        # Create colored image
        color_img = Image.new('RGB', (224, 224), color='red')
        color_path = cls.test_dir / "color.jpg"
        color_img.save(color_path)
        cls.test_images['color'] = str(color_path)
        
        # Create grayscale image
        gray_img = Image.new('L', (224, 224), color=128)
        gray_path = cls.test_dir / "gray.jpg"
        gray_img.save(gray_path)
        cls.test_images['gray'] = str(gray_path)

    def test_initialization(self):
        """Test detector initialization"""
        detector = NSFWDetector(confidence_threshold=0.7)
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertIsNotNone(detector.classifier)


    def test_detect_blank_image(self):
        """Test detection on blank image"""
        with Image.open(self.test_images['blank']) as img:
            result = self.detector.detect(img)
            
        self.assertIsInstance(result, NSFWDetectionResult)
        self.assertTrue(result.is_safe)
        self.assertEqual(result.category, NSFWCategory.SAFE)
        self.assertIsInstance(result.confidence, float)

    def test_detect_color_image(self):
        """Test detection on color image"""
        with Image.open(self.test_images['color']) as img:
            result = self.detector.detect(img)
            
        self.assertIsInstance(result, NSFWDetectionResult)
        self.assertIsInstance(result.is_safe, bool)
        self.assertIn(result.category, NSFWCategory)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_detect_grayscale_image(self):
        """Test detection on grayscale image"""
        with Image.open(self.test_images['gray']) as img:
            result = self.detector.detect(img)
            
        self.assertIsInstance(result, NSFWDetectionResult)
        self.assertIsInstance(result.is_safe, bool)

    def test_detect_invalid_image(self):
        """Test detection with invalid image"""
        invalid_img = Image.new('RGB', (1, 1))  # Too small image
        result = self.detector.detect(invalid_img)
        
        self.assertIsInstance(result, NSFWDetectionResult)
        self.assertTrue(result.is_safe)
        self.assertEqual(result.category, NSFWCategory.SAFE)

    def test_analyze_nudenet_result(self):
        """Test analysis of NudeNet results"""
        # Test explicit content
        explicit_result = [
            {'class': 'FEMALE_BREAST_EXPOSED',
            'score': 0.799403190612793,
            'box': [64, 182, 49, 51]},
            {'class': 'FACE_FEMALE',
            'score': 0.7881264686584473,
            'box': [82, 66, 36, 43]}
            ]
        result = self.detector._analyze_nudenet_result(explicit_result)
        self.assertEqual(result.category, NSFWCategory.EXPLICIT)
        self.assertFalse(result.is_safe)

        # Test suggestive content
        suggestive_result = [
            {'class': 'FEMALE_GENITALIA_COVERED',
            'score': 0.799403190612793,
            'box': [64, 182, 49, 51]},
            {'class': 'FACE_FEMALE',
            'score': 0.7881264686584473,
            'box': [82, 66, 36, 43]}
            ]
        result = self.detector._analyze_nudenet_result(suggestive_result)
        self.assertEqual(result.category, NSFWCategory.SUGGESTIVE)
        self.assertFalse(result.is_safe)

        # Test safe content
        safe_result = [
            {'class': 'FEMALE_BREAST_EXPOSED',
            'score': 0.1,
            'box': [64, 182, 49, 51]},
            {'class': 'FACE_FEMALE',
            'score': 0.7881264686584473,
            'box': [82, 66, 36, 43]}
            ]
        result = self.detector._analyze_nudenet_result(safe_result)
        self.assertEqual(result.category, NSFWCategory.SAFE)
        self.assertTrue(result.is_safe)

    def test_empty_result_handling(self):
        """Test handling of empty results"""
        result = self.detector._analyze_nudenet_result([])
        self.assertEqual(result.category, NSFWCategory.SAFE)
        self.assertTrue(result.is_safe)

    def test_invalid_result_handling(self):
        """Test handling of invalid results"""
        result = self.detector._analyze_nudenet_result([{"class": "invalid"}])
        self.assertEqual(result.category, NSFWCategory.SAFE)
        self.assertTrue(result.is_safe)

    def test_confidence_threshold_effect(self):
        """Test effect of different confidence thresholds"""
        # Create detectors with different thresholds
        strict_detector = NSFWDetector(confidence_threshold=0.9)
        lenient_detector = NSFWDetector(confidence_threshold=0.1)

        test_result = [
            {'class': 'FEMALE_BREAST_EXPOSED',
            'score': 0.5,
            'box': [64, 182, 49, 51]},
            {'class': 'FACE_FEMALE',
            'score': 0.8,
            'box': [82, 66, 36, 43]}
            ]

        # Test with strict threshold
        strict_result = strict_detector._analyze_nudenet_result(test_result)
        # Test with lenient threshold
        lenient_result = lenient_detector._analyze_nudenet_result(test_result)

        # Results should be different due to different thresholds
        self.assertNotEqual(strict_result.category, lenient_result.category)

    def test_multiple_detections(self):
        """Test multiple consecutive detections"""
        with Image.open(self.test_images['color']) as img:
            result1 = self.detector.detect(img)
            result2 = self.detector.detect(img)
            
        self.assertEqual(result1.category, result2.category)
        self.assertEqual(result1.is_safe, result2.is_safe)

if __name__ == '__main__':
    unittest.main(verbosity=2)
