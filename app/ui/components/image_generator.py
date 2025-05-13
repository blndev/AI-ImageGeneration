import os
import gradio as gr
import logging
from app.generators import FluxGenerator, GenerationParameters, ModelConfig, StabelDiffusionGenerator
from app.validators import PromptRefiner, NSFWDetector, CensorMethod, NSFWCategory
from app.utils.fileIO import save_image_with_timestamp, get_date_subfolder
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.analytics import Analytics
from .session_manager import SessionManager

# Set up module logger
logger = logging.getLogger(__name__)


class ImageGenerationHandler:
    def __init__(self, session_manager: SessionManager, config: AppConfig, analytics: Analytics, modelconfig: ModelConfig):
        self.config = config
        self.session_manager = session_manager
        self.analytics = analytics
        self.selectedmodelconfig = modelconfig

        self.nsfw_detector = NSFWDetector(confidence_threshold=0.7)

        self.initialize_image_generator()
        self.initialize_prompt_magic()
        self.MAX_NSFW_WARNINGS = -2  # amount of censored images if user generates nsfw content before we fully rewrite the prompt to avoid it

    def initialize_image_generator(self):
        if "flux" in self.selectedmodelconfig.model_type:
            self.generator = FluxGenerator(appconfig=self.config, modelconfig=self.selectedmodelconfig)
        else:
            self.generator = StabelDiffusionGenerator(appconfig=self.config, modelconfig=self.selectedmodelconfig)

    def initialize_prompt_magic(self):
        self.prompt_refiner = None
        self.promptmagic_enabled = False
        if self.config.feature_prompt_magic_enabled:
            self.prompt_refiner = PromptRefiner()
            self.promptmagic_enabled = self.prompt_refiner.validate_refiner_is_ready()

        if not self.promptmagic_enabled:
            logger.warning("NSFW protection via PromptMagic is turned off")

    def create_interface_elements(self, gr):
        with gr.Row():
            prompt = gr.Textbox(
                label="Your prompt",
                placeholder="Describe the image you want to generate...",
                value="",
                scale=4,
                lines=4,
                max_length=340
            )
            # ... other interface elements

        return prompt  # Return all created elements

    def generate_images(self,
                        session_state: SessionState,
                        prompt: str,
                        neg_prompt,
                        aspect_ratio: str,
                        user_activated_promptmagic: bool,
                        image_count):
        try:

            # cleanup input data
            prompt = str(prompt.strip()).replace("'", "-")
            userprompt = prompt
            neg_prompt = neg_prompt.strip()

            # enhance / shrink prompt and remove nsfw
            prompt = self._apply_prompt_magic(session_state, userprompt, user_activated_promptmagic)

            # additional nsfw protection
            if session_state.nsfw <= self.MAX_NSFW_WARNINGS:
                neg_prompt = "nude, naked, nsfw, porn," + neg_prompt

            logger.info(
                f"generating image for {session_state.session} with {session_state.token} credits available.\n - prompt: '{prompt}'")

            # split aspect ratio selection to dimensions by using modelconfig
            width, height = self._get_image_dimensions(aspect_ratio)

            generation_details = GenerationParameters(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(self.selectedmodelconfig.generation.get("steps", 30)),
                guidance_scale=float(self.selectedmodelconfig.generation.get("guidance", 1)),
                num_images_per_prompt=image_count,
                width=width,
                height=height
            )

            generated_images = self.generator.generate_images(params=generation_details)
            logger.debug(f"received {len(generated_images)} image(s) from generator")

            # reduce available credits after successful generation
            session_state.token -= image_count

            # apply censorship if still nsfw content is contained
            result_images = self._censor_nsfw_images(session_state, generated_images)

            # check saving output for validation of generation (Debug & Beta Only!!)
            self._save_output_for_debug(generation_details.to_dict(), userprompt, generated_images, result_images)

            return result_images, session_state, prompt

        except Exception as e:
            logger.error(f"image generation failed: {e}")
            logger.debug("Exception details:", exc_info=True)
            raise Exception("Error while generating the image")

    def _censor_nsfw_images(self, session_state, generated_images):
        result_images = []
        try:
            show_nsfw_censor_warning = False
            for image in generated_images:
                nsfw_check = self.nsfw_detector.detect(image)
                if not nsfw_check.is_safe:
                    # just for debugging purposes
                    logger.debug(f"Generated NSFW Image detected. Category: {nsfw_check.category}, Confidence: {nsfw_check.confidence}")

                # we check against nsfw<=0 to apply censorship on the explicit regions, if that happen more often (-2),
                # the prompt refiner is used in advance to avoid nsfw generation (trigger value is NSFW_WARNINGS)
                if not nsfw_check.is_safe and nsfw_check.category == NSFWCategory.EXPLICIT and session_state.nsfw <= 0:
                    result_images.append(
                        self.nsfw_detector.censor_detected_regions(
                            image=image,
                            detection_result=nsfw_check,
                            labels_to_censor=self.nsfw_detector.EXPLICIT_LABELS,
                            method=CensorMethod.PIXELATE)
                    )
                    show_nsfw_censor_warning = True
                    session_state.nsfw -= 1  # we must can go into negative values, as block starts after MAX_NSFW_WARNINGS limit
                else:
                    result_images.append(image)  # could be nsfw or not but it's allowed
                    if nsfw_check.category == NSFWCategory.EXPLICIT:
                        # reduce only if output is nsfw
                        session_state.nsfw -= 1
            if show_nsfw_censor_warning and self.config.feature_use_upload_for_age_check:
                gr.Info("""We censored at least one of your images.
                        You can remove the censorship by uploading images to train our System for better results or sharing Links.
                        Thanks for your understanding""", duration=0)
        except Exception as e:
            logger.warning(f"Error while NSFW check: {e}")
            result_images = generated_images
        return result_images

    def _apply_prompt_magic(self, session_state: SessionState, prompt: str, user_activated_promptmagic: bool) -> str:
        # check if nsfw or preview is allowed, enforce SFW prompt if not
        if session_state.nsfw <= self.MAX_NSFW_WARNINGS and self.prompt_refiner:
            if self.prompt_refiner.check_contains_nsfw(prompt):
                if self.config.feature_use_upload_for_age_check:
                    gr.Info("""Explicit image generation preview is over and will now be blocked. " \
                            You can get credits for uncensored images by uploading related images for our model training.""", duration=30)
                logger.info(f"Convert NSFW prompt to SFW. User Prompt: '{prompt}'")
                prompt = self.prompt_refiner.make_prompt_sfw(prompt, True)
            else:
                logger.debug("prompt is SFW")

        if self.prompt_refiner and user_activated_promptmagic:
            logger.info("Apply Prompt-Magic")
            # refine prompt multiple times for better reults
            for _ in range(3):
                prompt = self.prompt_refiner.magic_enhance(prompt, 200)
            if session_state.nsfw <= self.MAX_NSFW_WARNINGS and not self.prompt_refiner.is_safe_for_work(prompt):
                prompt = self.prompt_refiner.make_prompt_sfw(prompt)

        return prompt

    def _save_output_for_debug(self, gen_data: dict, userprompt: str, generated_images: list, result_images: list):
        # check saving output for validation of generation (Debug & Beta Only!!)
        if self.config.save_generated_output:
            logger.debug(f"saving images to {self.config.output_directory}")
            gen_data["userprompt"] = userprompt
            gen_data["model"] = self.selectedmodelconfig.model
            for image in generated_images:
                outdir = os.path.join(self.config.output_directory, get_date_subfolder(), "generation")
                save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True, generation_details=gen_data)

            for image in result_images:
                if image not in generated_images:
                    outdir = os.path.join(self.config.output_directory, get_date_subfolder(), "generation")
                    save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True,
                                              generation_details=gen_data, reference="result_")

    def _get_image_dimensions(self, aspect_ratio):
        width, height = 512, 512  # fallback
        for supported_ratio in self.selectedmodelconfig.aspect_ratio.keys():
            if supported_ratio.lower() in aspect_ratio.lower():
                ratio = self.selectedmodelconfig.aspect_ratio[supported_ratio]
                width, height = ModelConfig.split_aspect_ratio(ratio)
                break
        return width, height
