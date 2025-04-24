import gc, abc, os
from time import sleep
from typing import List
from PIL import Image, ImageDraw
import threading
from .modelconfig import ModelConfig
from ..appconfig import AppConfig
from . import GenerationParameters

# AI STuff
import torch

# Set up module logger
import logging
logger = logging.getLogger(__name__)


class ModelConfigException(Exception):
    pass


class BaseGenerator():
    def __init__(self, appconfig: AppConfig, modelconfig: ModelConfig):
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
        """Support function to check if the image is NSFW."""
        return x_image, False

    def warmup(self):
        """prepares the execution and load the model from storage or internet to the GPU"""
        self._load_model()

    def memory_optimization(self, pipeline):
        # cpu offload will not work with pipeline.to(cuda)
        if self.modelconfig.generation.get("GPU_ALLOW_ATTENTION_SLICING", False):
            logger.warning("attention slicing activated")
            pipeline.enable_attention_slicing(slice_size="auto")

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
        #  unload after a period of time without generation is triggered from ui, as this knows what will happen next
        try:
            logger.info("unload image generation model")
            if self._cached_generation_pipeline:
                del self._cached_generation_pipeline
                self._cached_generation_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.error("Error while unloading Image generation model")

    def _create_test_image(self, params: GenerationParameters) -> Image.Image:
        """Creates a debug image with white background and black text showing the parameters"""
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
    def _load_model(self, use_cached_model=True):
        pass
