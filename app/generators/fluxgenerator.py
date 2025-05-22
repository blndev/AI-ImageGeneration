
"""
Flux Generator Module

This module implements a Flux-based image generator that provides functionality for
generating images using the Flux AI model. It supports loading models from both
local safetensor files and remote sources, with automatic model caching and
memory optimization features.

The generator implements specialized handling for different Flux model variants
including 'dev' and 'schnell' configurations.

Classes:
    FluxGenerator: Singleton class implementing the BaseGenerator interface
    for Flux model implementations.

Dependencies:
    - torch: For GPU/CPU tensor operations
    - diffusers: For Flux pipeline implementations
    - PIL: For image processing
    - typing: For type hints
"""

from typing import List
from PIL import Image
from ..utils.singleton import singleton
from .modelconfig import ModelConfig
from .generation_params import GenerationParameters
from .base_generator import BaseGenerator, ModelConfigException

# AI STuff
import torch
from diffusers import FluxPipeline

# Set up module logger
import logging

logger = logging.getLogger(__name__)


@singleton
class FluxGenerator(BaseGenerator):
    """
    Flux implementation of the BaseGenerator interface.

    This class implements image generation using Flux models, supporting various
    model configurations and optimizations. It handles model loading, caching,
    and generation with specific parameter adjustments for different Flux variants.

    The class is implemented as a singleton to ensure only one instance of the
    model is loaded in memory at any time.

    Attributes:
        modelconfig (ModelConfig): Configuration for the current model
        appconfig (AppConfig): Application-wide configuration
    """

    def _load_model(self):
        """
        Load and configure the Flux Pipeline for image generation.

        This method handles the loading of Flux models from either local safetensor
        files or remote sources. It includes automatic model caching and applies
        appropriate memory optimizations.
        it is required that the self.modelconfig is proper set

        Returns:
            FluxPipeline: Configured Flux pipeline instance
            None: If loading fails

        Raises:
            ModelConfigException: If an unsupported model type is specified

        Notes:
            - Supports .safetensors files with from_single_file loading
            - Handles remote model loading via from_pretrained
            - Automatically configures device mapping and data types
            - Implements caching for improved performance

        """
        if self._cached_generation_pipeline:
            return self._cached_generation_pipeline

        try:
            modelpath = self.modelconfig.path
            logger.debug(f"Loading model {modelpath}, using cache '{self.appconfig.model_cache_dir}'")
            pipeline = None

            pipelinetype = None
            if "flux" in self.modelconfig.model_type.lower():
                pipelinetype = FluxPipeline

            if pipelinetype is None:
                raise ModelConfigException(
                    f"Unsupported model type for Flux Generator '{self.modelconfig.model_type}'. It must contain flux"
                )
            if modelpath.endswith("safetensors"):
                logger.info(
                    f"Using 'from_single_file' to load model {modelpath} from local folder"
                )
                pipeline = FluxPipeline.from_single_file(
                    modelpath,
                    token=self._hftoken,
                    local_files_only=True,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    use_safetensors=True,
                    device_map="auto",
                    add_watermarker=False,  # https://github.com/ShieldMnt/invisible-watermark/
                    strict=False,
                )
            else:
                logger.info(f"Using 'from_pretrained' option to load model {modelpath} from hugging face or local cache")
                pipeline = FluxPipeline.from_pretrained(
                    modelpath,
                    token=self._hftoken,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                )

            logger.debug("diffuser initiated")
            self._memory_optimization(pipeline)

            logger.debug("Flux-Pipeline created")
            self._cached_generation_pipeline = pipeline
            return pipeline
        except Exception as e:
            logger.error(f"Load Model failed. Error: {e}")
            # logger.debug("Exception details:", e)
            return None
            # raise Exception("Error while loading the pipeline for image conversion.\nSee logfile for details.")

    def change_model(self, modelconfig: ModelConfig):
        """
        Change the current generation model to a new one.

        Safely unloads the current model and loads a new one based on the
        provided configuration.

        Args:
            modelconfig (ModelConfig): Configuration for the new model

        Raises:
            Exception: If the modelconfig is invalid or loading fails
        """
        logger.info("Change generation model to %s", modelconfig)
        if type(modelconfig) is not type(ModelConfig):
            raise Exception("type of modelconfig must be ModelConfig")

        try:
            self.unload_model()
            self.modelconfig = modelconfig
            self._load_model()
        except Exception as e:
            logger.error(f"Error while changing text2img model: {e}")
            raise (f"Loading new img2img model '{self.modelconfig.model}' failed", e)

    def generate_images(self, params: GenerationParameters, status_callback) -> List[Image.Image]:
        """
        Generate images using the current Flux model.

        Handles the complete image generation process with specific adjustments
        for different Flux model variants (dev, schnell). Includes parameter
        validation, model loading, and embedding processing.

        Args:
            params (GenerationParameters): Parameters for image generation
                including prompts, number of images, steps, etc.

        Returns:
            ListImage.Imageof generated PIL Images

        Raises:
            Exception: If no model is loaded or if generation fails
            RuntimeError: If CUDA encounters memory issues

        Notes:
            - Automatically applies positive and negative embeddings
            - Supports special handling for 'dev' and 'schnell' model variants
            - Implements batch processing for multiple images
            - Thread-safe execution using generation lock
            - Handles NO_AI testing mode

        Example:
            ```python
            generator = FluxGenerator(...)
            params = GenerationParameters(
                prompt="an artistic landscape",
                num_images_per_prompt=1,
                num_inference_steps=30
            )
            images = generator.generate_images(params)
            ```
        """
        logger.debug("starting image generation")
        # Validate parameters and throw exceptions
        params.validate()
        if self.appconfig.NO_AI:
            logger.warning("'no ai' - option is activated")
            return self._create_test_image(params)

        logger.debug(
            f"start generating {params.num_images_per_prompt} images with {params.num_inference_steps} steps"
        )
        with self._generation_lock:
            try:
                current_pipeline = self._load_model()
                if not current_pipeline:
                    logger.error("No model loaded")
                    raise Exception("No model loaded. Generation not available")

                for embedding in self.modelconfig.embeddings["positive"]:
                    params.prompt = embedding.keyword + ", " + params.prompt

                for embedding in self.modelconfig.embeddings["negative"]:
                    params.negative_prompt = (
                        embedding.keyword + ", " + params.negative_prompt
                    )

                if "dev" in self.modelconfig.path.lower():
                    params = params.prepare_flux_dev()
                elif "schnell" in self.modelconfig.path.lower():
                    params = params.prepare_flux_schnell()

                logger.debug(
                    "Guidance: %f Strength: %f, Steps: %d",
                    params.guidance_scale,
                    params.strength,
                    params.num_inference_steps,
                )
                result_images = []
                imagecount = params.num_images_per_prompt
                params.num_images_per_prompt = 1
                for image in range(imagecount):
                    # TODO : add yield
                    result_images.append(current_pipeline(**params.to_dict()).images[0])
                    if status_callback:
                        status_callback(imagecount, image)
                return result_images

            except RuntimeError as e:
                logger.error(f"Error while generating Images: {e}")
                try:
                    torch.cuda.empty_cache()
                    self.unload_model()
                except Exception as e:
                    logger.debug(f"free cuda or unload model failed with {e}")
                raise Exception("Internal error while creating the image.")
