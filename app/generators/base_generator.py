"""
Base Generator Module for AI Image Generation

This module provides the foundation for AI-powered image generation implementations.
It handles basic GPU/CPU device management, model loading, and memory optimization
for AI image generation models.

Classes:
    ModelConfigException: Custom exception for model configuration errors
    BaseGenerator: Abstract base class for implementing AI image generators

Dependencies:
    - torch: For GPU/CPU tensor operations
    - PIL: For image processing
    - threading: For thread-safe model operations
    - logging: For application logging
"""

import gc
import abc
import os
from time import sleep
from typing import List
from PIL import Image, ImageDraw
import threading
from .modelconfig import ModelConfig
from ..appconfig import AppConfig
from . import GenerationParameters

import torch
import logging

logger = logging.getLogger(__name__)


class ModelConfigException(Exception):
    """
    Exception raised for errors in model configuration.

    This exception should be raised when there are issues with model initialization
    or configuration parameters.
    """
    pass


class BaseGenerator():
    """
    Abstract base class for AI image generation implementations.

    This class provides common functionality for AI image generators including
    device management, model loading/unloading, and memory optimization.

    Attributes:
        modelconfig (ModelConfig): Configuration for the AI model
        appconfig (AppConfig): Application-wide configuration
        device (str): Computing device ('cuda' or 'cpu')
        torch_dtype: PyTorch data type for tensor operations
        _cached_generation_pipeline: Cached model pipeline (defined by concrete implementations)
        _generation_lock (threading.Lock): Thread lock for generation operations
        _hftoken (str): HuggingFace API token

    Args:
        appconfig (AppConfig): Application configuration instance
        modelconfig (ModelConfig): Model configuration instance
    """

    def __init__(self, appconfig: AppConfig, modelconfig: ModelConfig):
        """
        Initialize the generator with application and model configurations.

        Args:
            appconfig (AppConfig): Application configuration instance
            modelconfig (ModelConfig): Model configuration instance
        """
        logger.info("Initialize Base Generator")
        self.modelconfig = modelconfig
        if self.modelconfig:
            logger.info(f"selected model: '{self.modelconfig.model}'")

        self.appconfig = appconfig

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Set device for generation to {self.device}")

        self._cached_generation_pipeline = None
        self._generation_lock = threading.Lock()

        logger.info(f"using cache directory '{self.appconfig.model_cache_dir}'")
        try:
            os.makedirs(self.appconfig.model_cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"failed to create the model directory {e}")

        self._hftoken = os.getenv("HUGGINGFACE_TOKEN", None)
        if self._hftoken:
            logger.info("Huggingface Tokem provided")

    def __del__(self):
        logger.info("free memory used for Generator pipeline")
        self.unload_model()

    def check_safety(self, x_image):
        """
        Check if the generated image is safe for viewing (non-NSFW).

        Args:
            x_image: Image to be checked

        Returns:
            tuple: (image, is_nsfw_flag) - true if NSFW
        """
        # implemented in nsfw_check in image_generator, because of advanced logic
        return x_image, False

    def warmup(self) -> None:
        """
        Prepare the model for execution by loading it into memory.
        This method should be called before generating images to ensure
        the model is ready for inference.
        """
        self._load_model()

    def _memory_optimization(self, pipeline):
        """
        Apply memory optimization techniques to the model pipeline.

        Configures various optimization settings like attention slicing,
        xformers optimization, and GPU memory management based on the
        model configuration.

        Args:
            pipeline: The model pipeline to optimize
        """
        # cpu offload will not work with pipeline.to(cuda)
        if self.modelconfig.generation.get("GPU_ALLOW_ATTENTION_SLICING", False):
            logger.warning("attention slicing activated")
            pipeline.enable_attention_slicing(slice_size="auto")

        # FIXME TODO BUG seems not to work to read this, must be off for flux!!!
        if self.modelconfig.generation.get("GPU_ALLOW_XFORMERS", False):
            logger.info("xformers activated")
            pipeline.enable_xformers_memory_efficient_attention()

        if self.device == "cuda" and not self.modelconfig.generation.get("GPU_ALLOW_MEMORY_OFFLOAD", False):
            logger.info("checking embeddings")
            torch.cuda.empty_cache()

            try:
                # both vars used later, import to have them as None
                self.embedding_positive = None
                self.embedding_negative = None

                for embedding in self.modelconfig.embeddings["positive"]:
                    pipeline.load_textual_inversion(
                        embedding.source, token=embedding.keyword
                    )

                for embedding in self.modelconfig.embeddings["negative"]:
                    pipeline.load_textual_inversion(
                        embedding.source, token=embedding.keyword
                    )

            except Exception as e:
                logger.error(f"Loading embeddings failed {e}")

        if self.modelconfig.generation.get("GPU_ALLOW_MEMORY_OFFLOAD", False):
            logger.warning("gpu offload activated")
            pipeline.enable_model_cpu_offload()
        else:
            logger.info(f"load model to {self.device}")
            pipeline.to(self.device)

        self._log_gpu_memory_usage()

    def _log_gpu_memory_usage(self):
        """
        Log current GPU memory usage statistics.

        Logs total, allocated, and cached GPU memory if CUDA is available.
        This method is used for debugging and monitoring purposes.
        """
        if self.device == "cuda" and torch.cuda.is_available():
            # Get the GPU ID
            gpu_id = torch.cuda.current_device()

            # Get the total and allocated memory
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            cached_memory = torch.cuda.memory_reserved(gpu_id)

            logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
            logger.info(f"Allocated GPU memory: {allocated_memory / 1024**3:.2f} GB")
            logger.info(f"Cached GPU memory: {cached_memory / 1024**3:.2f} GB")
        else:
            pass

    def unload_model(self):
        """
        Unload the model from memory and free up GPU resources.

        This method should be called when the model is no longer needed
        to free up system resources.
        """
        try:
            if self._cached_generation_pipeline:
                logger.info("unload image generation model")
                del self._cached_generation_pipeline
                self._cached_generation_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.error("Error while unloading Image generation model")

    def _create_test_image(self, params: GenerationParameters) -> Image.Image:
        """
        Create test images with parameter information for debugging.

        Args:
            params (GenerationParameters): Parameters for image generation

        Returns:
            List[Image.Image]: List of generated test images with parameter text
        """
        images = []
        bg_colors = [
            "aliceblue",
            "lightgoldenrodyellow",
            "lightcyan",
            "mistyrose",
            "honeydew",
            "beige",
            "lavenderblush",
            "azure",
            "ghostwhite",
            "floralwhite",
        ]

        for i in range(params.num_images_per_prompt):
            sleep(5)
            # Create a new white image
            img = Image.new(
                "RGB",
                (params.width, params.height),
                color=bg_colors[i] if len(bg_colors) > i else "white",
            )

            # Create draw object
            draw = ImageDraw.Draw(img)

            # Convert parameters to text
            param_dict = params.to_dict()
            text_lines = []
            text_lines.append(f"Prompt: {param_dict['prompt']}")
            text_lines.append(f"Steps: {param_dict['num_inference_steps']}")
            text_lines.append(f"Guidance Scale: {param_dict['guidance_scale']}")
            text_lines.append(f"Size: {param_dict['width']}x{param_dict['height']}")
            if "negative_prompt" in param_dict:
                text_lines.append(f"Negative Prompt: {param_dict['negative_prompt']}")
            if "generator" in param_dict:
                text_lines.append(f"Seed: {params.seed}")
            if "image" in param_dict:
                text_lines.append(f"Strength: {param_dict['strength']}")

            # Draw text
            y_position = 20
            for line in text_lines:
                draw.text((20, y_position), line, fill="black")
                y_position += 30
            images.append(img)
        return images

    @abc.abstractmethod
    def _load_model(self, use_cached_model: bool = True) -> None:
        """
        Abstract method to load the AI model.

        This method must be implemented by concrete subclasses to define
        how their specific model should be loaded.

        Args:
            use_cached_model (bool): Whether to use a cached model if available

        Raises:
            NotImplementedError: If not implemented by concrete subclass
        """
        pass

    @abc.abstractmethod
    def generate_images(self, params: GenerationParameters, status_callback) -> List[Image.Image]:
        pass
