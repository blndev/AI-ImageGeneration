
"""
Stable Diffusion Generator Module

This module implements a Stable Diffusion-based image generator that supports both
standard Stable Diffusion 1.5 and SDXL models. It provides functionality for
generating images from text prompts using different model architectures.

The generator supports both local model files (safetensors) and models from
Hugging Face, with automatic model caching and memory optimization features.

Classes:
    StabelDiffusionGenerator: Singleton class implementing the BaseGenerator
    interface for Stable Diffusion models.

Dependencies:
    - torch: For GPU/CPU tensor operations
    - diffusers: For Stable Diffusion pipeline implementations
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
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# Set up module logger
import logging

logger = logging.getLogger(__name__)


# TODO fo SDXL use refiner as well: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
@singleton
class StabelDiffusionGenerator(BaseGenerator):
    """
    Stable Diffusion implementation of the BaseGenerator interface.

    This class implements image generation using either Stable Diffusion 1.5 or
    SDXL models. It supports loading models from local safetensor files or
    from Hugging Face, with automatic caching and memory optimization.

    The class is implemented as a singleton to ensure only one instance of the
    model is loaded in memory at any time.

    Attributes:
        modelconfig (ModelConfig): Configuration for the current model
        appconfig (AppConfig): Application-wide configuration
    """

    def _load_model(self):
        """
        Load and configure the Stable Diffusion Pipeline for image generation.

        Requires self.modelconfig is set!

        This method handles the loading of different types of Stable Diffusion
        models (1.5 or SDXL) from either local safetensor files or Hugging Face.
        It includes automatic model caching and memory optimization.

        Returns:
            StableDiffusionPipeline or StableDiffusionXLPipeline: Configured pipeline
            None: If loading fails

        Raises:
            ModelConfigException: If an unsupported model type is specified

        Notes:
            - Supports both .safetensors files and Hugging Face models
            - Automatically handles device placement and memory optimization
        """
        if self._cached_generation_pipeline:
            return self._cached_generation_pipeline

        try:
            modelpath = self.modelconfig.path
            logger.debug(f"Loading model {modelpath}, using cache '{self.appconfig.model_cache_dir}'")
            pipeline = None

            pipelinetype = None
            if "1.5" in self.modelconfig.model_type:
                pipelinetype = StableDiffusionPipeline
            elif "sdxl" in self.modelconfig.model_type.lower():
                pipelinetype = StableDiffusionXLPipeline

            if pipelinetype is None:
                raise ModelConfigException(
                    f"Unsupported model type for StabelDiffusion Generator '{self.modelconfig.model_type}'. It must contain 1.5 or sdxl"
                )
            if modelpath.endswith("safetensors"):
                logger.info(
                    f"Using 'from_single_file' to load model {modelpath} from local folder"
                )
                pipeline = pipelinetype.from_single_file(
                    modelpath,
                    token=self._hftoken,
                    local_files_only=True,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    device_map="auto",
                    add_watermarker=False,  # https://github.com/ShieldMnt/invisible-watermark/
                    strict=False,
                )
            else:
                logger.info(
                    f"Using 'from_pretrained' option to load model {modelpath} from hugging face or local cache"
                )
                pipeline = pipelinetype.from_pretrained(
                    modelpath,
                    token=self._hftoken,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                )

            logger.debug("diffuser initiated")

            self._memory_optimization(pipeline)

            #     # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
            #     pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            #     logger.info("torch compile done")

            # pipeline.enable_attention_slicing("auto")
            #
            logger.debug("Diffusion-Pipeline created")
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

        This method safely unloads the current model and loads a new one
        based on the provided configuration.

        Args:
            modelconfig (ModelConfig): Configuration for the new model

        Raises:
            Exception: If the modelconfig is not of type ModelConfig
            Exception: If loading the new model fails
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

    def generate_images(self, params: GenerationParameters) -> List[Image.Image]:
        """
        Generate images using the current Stable Diffusion model.

        This method handles the complete image generation process, including:
        - Parameter validation
        - Model loading and verification
        - Prompt preprocessing with embeddings
        - Batch image generation
        - Error handling and resource cleanup

        Args:
            params (GenerationParameters): Parameters for image generation
                including prompts, number of images, steps, etc.

        Returns:
            List[Image.Image]: List of generated PIL Images

        Raises:
            Exception: If no model is loaded or if generation fails
            RuntimeError: If CUDA encounters memory issues

        Notes:
            - Handles both SD 1.5 and SDXL model types
            - Automatically applies positive and negative embeddings
            - Implements batch processing for multiple images
            - Includes CUDA memory management
            - Thread-safe execution using generation lock

        Example:
            ```python
            generator = StabelDiffusionGenerator(...)
            params = GenerationParameters(
                prompt="a beautiful sunset",
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

                if "1.5" in self.modelconfig.model_type.lower():
                    params = params.prepare_stablediffusion_std()
                elif "sdxl" in self.modelconfig.model_type.lower():
                    params = params.prepare_stablediffusion_std()

                for embedding in self.modelconfig.embeddings["positive"]:
                    params.prompt = embedding.keyword + ", " + params.prompt

                for embedding in self.modelconfig.embeddings["negative"]:
                    params.negative_prompt = (
                        embedding.keyword + ", " + params.negative_prompt
                    )

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
                return result_images

            except RuntimeError as e:
                logger.error(f"Error while generating Images: {e}")
                try:
                    torch.cuda.empty_cache()
                    self.unload_model()
                except Exception as e:
                    logger.debug(f"free cuda or unload model failed with {e}")
                raise Exception("Internal error while creating the image.")
