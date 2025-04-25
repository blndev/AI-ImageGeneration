
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
    # def __init__(self, appconfig:AppConfig, modelconfig:ModelConfig):
    #     logger.info("Initialize StabelDiffusion Generator")

    def _load_model(self):
        """Load and return the Pipeline to generate images"""
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
            self.memory_optimization(pipeline)

            logger.debug("Flux-Pipeline created")
            self._cached_generation_pipeline = pipeline
            return pipeline
        except Exception as e:
            logger.error(f"Load Model failed. Error: {e}")
            # logger.debug("Exception details:", e)
            return None
            # raise Exception("Error while loading the pipeline for image conversion.\nSee logfile for details.")

    def change_model(self, modelconfig: ModelConfig):
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
        """Generate a list of images."""
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
                return result_images

            except RuntimeError as e:
                logger.error(f"Error while generating Images: {e}")
                try:
                    torch.cuda.empty_cache()
                    self.unload_model()
                except Exception as e:
                    logger.debug(f"free cuda or unload model failed with {e}")
                raise Exception("Internal error while creating the image.")
