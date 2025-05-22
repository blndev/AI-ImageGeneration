from PIL import Image, ImageFilter
import numpy as np
from typing import Dict, Union, List, Set
import logging
from enum import Enum
import os
from dataclasses import dataclass
from nudenet import NudeDetector as nndetector
import tempfile


class NSFWCategory(Enum):
    """Categories for NSFW detection"""
    SAFE = "safe"
    SUGGESTIVE = "suggestive"
    EXPLICIT = "explicit"
    UNKNOWN = "unknown"

class CensorMethod(Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK = "black"
    WHITE = "white"

@dataclass
class NSFWDetectionResult:
    """Result of NSFW detection"""
    is_safe: bool
    category: NSFWCategory
    confidence: float
    details: Dict[str, float]


class NSFWDetector:
    """
    NSFW content detector using NudeNet

    Example
    # result = detector.detect(img)

    # Log results
    logging.info(f"Category: {result.category.value}")
    logging.info(f"Confidence: {result.confidence:.2f}")
    logging.info(f"Details: {result.details}")

    # return result.is_safe
    """

    # Define label sets as class attributes
    EXPLICIT_LABELS = {
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED"
    }

    SUGGESTIVE_LABELS = {
        "FEMALE_GENITALIA_COVERED",
        # "BUTTOCKS_COVERED",
        # "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "ARMPITS_EXPOSED"
    }

    def __init__(self,
                 confidence_threshold: float = 0.5,
                 model_path: str = None):
        """
        Initialize the NSFW detector

        Args:
            confidence_threshold: Threshold for NSFW detection (0.0 to 1.0)
            model_path: Optional path to custom model file
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.classifier = None

        try:
            self._init_model(model_path)
        except Exception as e:
            self.logger.error(f"Error initializing NSFW detector: {str(e)}")
            raise

    def _init_model(self, model_path: str = None) -> None:
        """Initialize NudeNet model"""
        try:
            if model_path:
                self.classifier = nndetector(model_path=model_path)
            else:
                self.classifier = nndetector()
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise

    def _save_image_temporarily(self, image: Image.Image) -> str:
        """
        Save image to temporary file

        Args:
            image: PIL Image to save

        Returns:
            str: Path to temporary file
        """
        try:
            # Create temporary file with .jpg extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name

            # Save image in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(temp_path, 'JPEG')

            return temp_path

        except Exception as e:
            self.logger.error(f"Error saving temporary image: {str(e)}")
            raise

    def _analyze_nudenet_result(self, classification: Dict[str, float]) -> NSFWDetectionResult:
        """
        Analyze NudeNet classification result

        Args:
            result: Dictionary containing NudeNet classification scores for each label

        Returns:
            NSFWDetectionResult object
        """
        try:
 
            explicit_score = 0
            suggestive_score = 0
            if len(classification) > 0:
                # Calculate scores by finding matching classes and summing their scores
                explicit_score = sum(
                    item['score'] 
                    for item in classification 
                    if item['class'] in self.EXPLICIT_LABELS
                )
                suggestive_score = sum(
                    item['score'] 
                    for item in classification 
                    if item['class'] in self.SUGGESTIVE_LABELS
                )

            # Determine category and confidence
            if explicit_score > self.confidence_threshold:
                category = NSFWCategory.EXPLICIT
                confidence = explicit_score
            elif suggestive_score > self.confidence_threshold:
                category = NSFWCategory.SUGGESTIVE
                confidence = suggestive_score
            else:
                category = NSFWCategory.SAFE
                confidence = 1.0 - max(explicit_score, suggestive_score)

            return NSFWDetectionResult(
                is_safe=category == NSFWCategory.SAFE,
                category=category,
                confidence=confidence,
                details=classification
            )

        except Exception as e:
            self.logger.error(f"Error analyzing results: {str(e)}")
            return NSFWDetectionResult(
                is_safe=False,
                category=NSFWCategory.UNKNOWN,
                confidence=0.0,
                details={'error': str(e)}
            )

    def detect(self, image: Image.Image) -> NSFWDetectionResult:
        """
        Detect NSFW content in an image

        Args:
            image: PIL Image to analyze

        Returns:
            NSFWDetectionResult object
        """
        if not self.classifier:
            raise RuntimeError("Model not initialized")

        temp_path = None
        try:
            # Save image temporarily
            temp_path = self._save_image_temporarily(image)
            classification = self.classifier.detect(temp_path)
            # Analyze results
            return self._analyze_nudenet_result(classification)

        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return NSFWDetectionResult(
                is_safe=False,
                category=NSFWCategory.UNKNOWN,
                confidence=0.0,
                details={'error': str(e)}
            )
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Error removing temporary file: {str(e)}")


    def _apply_censoring(
        self, 
        region: Image.Image, 
        method: CensorMethod, 
        **kwargs
    ) -> Image.Image:
        """
        Apply censoring method to image region

        Args:
            region: Image region to censor
            method: Censoring method to apply
            **kwargs: Additional parameters for censoring method

        Returns:
            Censored image region
        """
        try:
            if method == CensorMethod.BLUR:
                return region.filter(ImageFilter.GaussianBlur(
                    radius=kwargs.get('blur_radius', 30)
                ))
            elif method == CensorMethod.PIXELATE:
                small = region.resize(
                    (region.width // 16, region.height // 16),
                    Image.Resampling.BILINEAR
                )
                return small.resize(
                    region.size,
                    Image.Resampling.NEAREST
                )
            elif method == CensorMethod.BLACK:
                return Image.new('RGB', region.size, 'black')
            elif method == CensorMethod.WHITE:
                return Image.new('RGB', region.size, 'white')
            else:
                return region.filter(ImageFilter.GaussianBlur(radius=30))
        except Exception as e:
            self.logger.error(f"Error applying censoring: {str(e)}")
            return region

    def censor_detected_regions(
        self,
        image: Image.Image,
        detection_result: NSFWDetectionResult,
        labels_to_censor: Set[str] = None,
        method: CensorMethod = CensorMethod.BLUR,
        padding: int = 0,
        **kwargs
    ) -> Image.Image:
        """
        Censor detected regions in image

        Args:
            image: PIL Image to censor
            detection_result: Detection result from detect()
            labels_to_censor: Set of labels to censor (if None, uses EXPLICIT_LABELS)
            method: Censoring method to apply
            padding: Padding around detected regions
            **kwargs: Additional parameters for censoring method

        Returns:
            Censored image
        """
        try:
            # If no specific labels provided, use explicit labels
            if labels_to_censor is None:
                labels_to_censor = self.EXPLICIT_LABELS

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Create copy of image to modify
            censored_image = image.copy()

            # Process each detection
            for detection in detection_result.details:
                # Skip if class not in labels to censor
                if detection['class'] not in labels_to_censor:
                    continue

                # Extract and adjust box coordinates
                x, y, width, height = detection['box']
                left = max(0, x - padding)
                top = max(0, y - padding)
                right = min(image.width, x + width + padding)
                bottom = min(image.height, y + height + padding)

                # Extract region
                region = censored_image.crop((left, top, right, bottom))

                # Apply censoring
                censored_region = self._apply_censoring(
                    region, 
                    method, 
                    **kwargs
                )

                # Paste back
                censored_image.paste(censored_region, (left, top))

            return censored_image

        except Exception as e:
            self.logger.error(f"Error censoring image: {str(e)}")
            return image

def create_nsfw_detector(confidence_threshold: float = 0.8) -> NSFWDetector:
    """
    Factory function to create NSFW detector

    Args:
        confidence_threshold: Threshold for NSFW detection

    Returns:
        NSFWDetector instance
    """
    return NSFWDetector(confidence_threshold=confidence_threshold)
