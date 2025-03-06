import gc
import os
from typing import List
from distutils.util import strtobool
from PIL import Image, ImageDraw
import logging
import threading
from app.utils.singleton import singleton
from app import FluxParameters

#AI STuff
import torch
from diffusers import FluxPipeline, StableDiffusionXLPipeline

# Set up module logger
logger = logging.getLogger(__name__)

@singleton
class FluxGenerator():
    def __init__(self, sdxl: bool = False):
        logger.info("Initialize FluxGenerator")
        #black-forest-labs/FLUX.1-dev
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Set device for generation to {self.device}")

        self._cached_generation_pipeline = None
        self._generation_lock = threading.Lock()

        if sdxl or bool(strtobool(os.getenv("USE_SDXL", "False"))):
            self.pipelinetemplate = StableDiffusionXLPipeline
            logger.info("using SDXL")
            self.SDXL = True
        else:
            logger.info("using FLUX")
            self.pipelinetemplate = FluxPipeline
            self.SDXL=False

        self.model_directory = os.getenv("MODEL_DIRECTORY", "./models/")
        logger.info(f"using cache directory '{self.model_directory}'")
        try:
            os.makedirs(self.model_directory , exist_ok=True)
        except Exception as e:
            logger.warning(f"failed to create the model directory {e}")

        self.model = os.getenv("GENERATION_MODEL", "black-forest-labs/FLUX.1-dev")
        self._hftoken = os.getenv("HUGGINGFACE_TOKEN", None)

    def __del__(self):
        logger.info("free memory used for FluxGenerator pipeline")
        self.unload_model()

    def _download_model():
        """will download missing parts of the model if huggingface token is not provided"""
        #TODO: implement
        pass
        #https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors?download=true
        #https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e5m2.safetensors?download=true

    def check_safety(self, x_image):
        """Support function to check if the image is NSFW."""
        return x_image, False

    def warmup(self):
        """prepares the execution and load the model from storage or internet to the GPU"""
        self._load_model()

    def _load_model(self, use_cached_model=True):
        """Load and return the Stable Diffusion model to generate images"""
        if self._cached_generation_pipeline and use_cached_model:
            return self._cached_generation_pipeline

        try:
            logger.debug(f"Loading model {self.model}, using cache  '{self.model_directory}'")
            pipeline = None
            
            if self.model.endswith("safetensors"):
                logger.info(f"Using 'from_single_file' to load model {self.model} from local folder")
                pipeline = self.pipelinetemplate.from_single_file(
                    self.model,
                    token = self._hftoken,
                    local_files_only=True,
                    cache_dir = self.model_directory,
                    torch_dtype=self.torch_dtype,
                    #safety_checker=None,
                    #requires_safety_checker=False,
                    use_safetensors=True,
                    device_map="auto",
                    strict=False
                )
            else:
                logger.info(f"Using 'from_pretrained' option to load model {self.model} from hugging face or local cache")
                pipeline = self.pipelinetemplate.from_pretrained(
                    self.model,
                    token=self._hftoken,
                    cache_dir = self.model_directory,
                    torch_dtype=self.torch_dtype,
                    #safety_checker=None,
                    #requires_safety_checker=False
                )

            logger.debug("diffuser initiated")
            
            #cpu offload will not work with pipeline.to(cuda)
            if bool(strtobool(os.getenv("GPU_ALLOW_ATTENTION_SLICING", "False"))):
                logger.info("attention slicing activated")
                pipeline.enable_attention_slicing(slice_size="auto")

            if bool(strtobool(os.getenv("GPU_ALLOW_XFORMERS", "False"))):
                logger.info("xformers activated") 
                pipeline.enable_xformers_memory_efficient_attention()

            if self.device=="cuda":
                torch.cuda.empty_cache()

            if bool(strtobool(os.getenv("GPU_ALLOW_MEMORY_OFFLOAD", "False"))):
                logger.warning("gpu offload activated")
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(self.device)

            #pipeline.enable_attention_slicing("auto")
            #
            logger.debug("FluxGenerator-Pipeline created")
            self._cached_generation_pipeline = pipeline
            return pipeline
        except Exception as e:
            logger.error(f"Load Model failed. Error: {e}")
            #logger.debug("Exception details:", e)
            return None
            #raise Exception("Error while loading the pipeline for image conversion.\nSee logfile for details.")

    def unload_model(self):
        try:
            logger.info("unload image generation model")
            if self._cached_generation_pipeline:
                del self._cached_generation_pipeline
                self._cached_generation_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.error("Error while unloading Image generation model")

    def change_model(self, model):
        logger.info("Change generation model to %s", model)
        try:
            self.default_model = model
            self.unload_model()
            self._load_model(model=model, use_cached_model=False)
        except Exception as e:
            logger.error("Error while changing text2img model: %s", str(e))
            logger.debug("Exception details: {e}")
            raise (f"Loading new img2img model '{model}' failed", e)

    def is_flux_schnell(self):
        """true if a schnell model is used (requires different generation parameters than dev)"""
        return "schnell" in self.model.lower()
    
    def is_flux_dev(self):
        """true if a dev model is used"""
        return "dev" in self.model.lower()
    
    def generate_images(self, params: FluxParameters) -> List[Image.Image]:
        """Generate a list of images."""
        logger.debug("starting image generation")
        # Validate parameters and throw exceptions
        params.validate()
        if os.getenv("NO_AI", False):
            logger.warning("'no ai' - option is activated")
            return self._create_debug_image(params)
        
        with self._generation_lock:
            try:
                model = self._load_model()
                if not model:
                    logger.error("No model loaded")
                    raise Exception("No model loaded. Generation not available")

                #https://huggingface.co/docs/diffusers/main/api/pipelines/flux
                if self.is_flux_schnell():
                    params = params.prepare_flux_schnell()
                elif self.is_flux_dev():
                    params = params.prepare_flux_dev()
                elif self.SDXL:
                    params = params.prepare_sdxl()

                logger.debug("Strength: %f, Steps: %d", params.strength, params.num_inference_steps)
                result_images = model(**params.to_dict()).images
                return result_images

            except RuntimeError as e:
                logger.error(f"Error while generating Images: {e}")
                try:
                    pass#self.unload_model()
                except:
                    pass
                raise Exception(f"Internal error while creating the image.")


    def _create_debug_image(self, params: FluxParameters) -> Image.Image:
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
            "floralwhite"
        ]
        
        for i in range(params.num_images_per_prompt):
            # Create a new white image
            img = Image.new('RGB', (params.width, params.height), color=bg_colors[i] if len (bg_colors)>i else "white")

            # Create draw object
            draw = ImageDraw.Draw(img)
            
            # Convert parameters to text
            param_dict = params.to_dict()
            text_lines = []
            text_lines.append(f"Prompt: {param_dict['prompt']}")
            text_lines.append(f"Steps: {param_dict['num_inference_steps']}")
            text_lines.append(f"Guidance Scale: {param_dict['guidance_scale']}")
            text_lines.append(f"Size: {param_dict['width']}x{param_dict['height']}")
            if 'negative_prompt' in param_dict:
                text_lines.append(f"Negative Prompt: {param_dict['negative_prompt']}")
            if 'generator' in param_dict:
                text_lines.append(f"Seed: {params.seed}")
            if 'image' in param_dict:
                text_lines.append(f"Strength: {param_dict['strength']}")
            
            # Draw text
            y_position = 20
            for line in text_lines:
                draw.text((20, y_position), line, fill='black')
                y_position += 30
            images.append(img)            
        return images

