
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
    # def __init__(self, appconfig:AppConfig, modelconfig:ModelConfig):
    #     logger.info("Initialize StabelDiffusion Generator")

    def _load_model(self):
        """Load and return the Stable Diffusion model to generate images"""
        if self._cached_generation_pipeline:
            return self._cached_generation_pipeline

        try:
            modelpath = self.modelconfig.path
            logger.debug(f"Loading model {modelpath}, using cache '{self.appconfig.model_cache_dir}'")
            pipeline = None

            pt = None
            if "1.5" in self.modelconfig.model_type:
                pt = StableDiffusionPipeline
            elif "sdxl" in self.modelconfig.model_type.lower():
                pt = StableDiffusionXLPipeline

            if pt is None:
                raise ModelConfigException(
                    f"Unsupported model type for StabelDiffusion Generator '{self.modelconfig.model_type}'. It must contain 1.5 or sdxl"
                )
            if modelpath.endswith("safetensors"):
                logger.info(
                    f"Using 'from_single_file' to load model {modelpath} from local folder"
                )
                pipeline = pt.from_single_file(
                    modelpath,
                    token=self._hftoken,
                    local_files_only=True,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,  # TODO: use function callback with llava
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
                pipeline = pt.from_pretrained(
                    modelpath,
                    token=self._hftoken,
                    cache_dir=self.appconfig.model_cache_dir,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,  # TODO: use function callback with llava
                    requires_safety_checker=False,
                    device_map="auto",
                )

            logger.debug("diffuser initiated")

            self.memory_optimization(pipeline)

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
        logger.info("Change generation model to %s", modelconfig)
        # TODO: check if the type comparison is working
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

                # https://huggingface.co/docs/diffusers/main/api/pipelines/flux
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
                    self._unload_model()
                except Exception as e:
                    logger.debug(f"free cuda or unload model failed with {e}")
                raise Exception("Internal error while creating the image.")
