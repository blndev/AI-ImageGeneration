import os
import random
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

    def __status_callback(self, total_images, current_image):
        try:
            current_image += 1
            current_progress = 0.15 + (current_image * 0.65 / total_images)
            if current_progress < 0 or current_progress > 1: current_progress = 0.5
            if total_images == 1: current_progress = 0.5
            logger.debug(f"Create image {current_image} of {total_images}. Current Progress: {current_progress * 100}%")
            if type(self.gradio_progress_callback) is gr.helpers.Progress:
                gradio_progress = self.gradio_progress_callback
                gradio_progress(progress=current_progress, desc=f"Generating image {current_image} of {total_images}")
            else:
                logger.debug(f"No progress callback available. Create image {current_image} of {total_images}")
        except Exception as e:
            logger.error(f"Status Callback error: {e}")

    def generate_images(self,
                        progress: gr.Progress,
                        session_state: SessionState,
                        prompt: str,
                        neg_prompt,
                        aspect_ratio: str,
                        user_activated_promptmagic: bool,
                        image_count: int):
        try:

            # cleanup input data
            prompt = str(prompt.strip()).replace("'", "-")
            userprompt = prompt
            style = None
            try:
                if self.config.promptmarker in prompt:
                    parts = prompt.split(self.config.promptmarker)
                    if len(parts) > 1:
                        userprompt = parts[1]
                        style = parts[0] + "{userprompt}"
                    if len(parts) > 2:
                        style = style + parts[2]
                    logger.debug(f"Style '{style}' identified. Style free prompt: '{userprompt}'")
            except Exception:
                logger.info("error while extracting style from prompt")

            neg_prompt = neg_prompt.strip()

            progress(0.1, "analyze prompt")
            # enhance / shrink prompt and remove nsfw
            prompt = self._apply_prompt_magic(session_state, userprompt, user_activated_promptmagic)
            # reapply style if exists
            if style:
                prompt = style.replace("{userprompt}", prompt)

            # additional nsfw protection
            if session_state.nsfw <= self.MAX_NSFW_WARNINGS:
                neg_prompt = "nude, naked, nsfw, porn," + neg_prompt

            logger.info(
                f"""generating image for {session_state.session}
                with {session_state.token} credits ({session_state.nsfw} NSFW) available
                prompt: '{prompt}'""")

            # split aspect ratio selection to dimensions by using modelconfig
            width, height = self._get_image_dimensions(aspect_ratio)
            progress(0.15, "load ai system")

            generation_details = GenerationParameters(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(self.selectedmodelconfig.generation.get("steps", 30)),
                guidance_scale=float(self.selectedmodelconfig.generation.get("guidance", 1)),
                num_images_per_prompt=image_count,
                width=width,
                height=height,
            )
            self.gradio_progress_callback = progress
            generated_images = self.generator.generate_images(params=generation_details, status_callback=self.__status_callback)
            logger.debug(f"received {len(generated_images)} image(s) from generator")
            self.gradio_progress_callback = None

            # reduce available credits after successful generation
            session_state.token -= image_count

            progress(0.9, "validate generated images")
            # apply censorship if still nsfw content is contained
            result_images = self._censor_nsfw_images(session_state, generated_images)

            # check saving output for validation of generation (Debug & Beta Only!!)
            self._save_output_for_debug(gen_data=generation_details.to_dict(),
                                        userprompt=userprompt,
                                        generated_images=generated_images,
                                        result_images=result_images
                                        )

            return result_images, session_state, prompt

        except Exception as e:
            logger.error(f"image generation failed: {e}")
            logger.debug("Exception details:", exc_info=True)
            self.analytics.record_application_error(module="image generation", criticality="error")
            raise Exception("Error while generating the image")

    def _censor_nsfw_images(self, session_state, generated_images):
        result_images = []
        try:
            show_nsfw_censor_warning = False
            nsfw_count = 0
            for image in generated_images:
                nsfw_check = self.nsfw_detector.detect(image)
                if not nsfw_check.is_safe:
                    # just for debugging purposes
                    logger.debug(f"Generated NSFW Image detected. Category: {nsfw_check.category}, Confidence: {nsfw_check.confidence}")
                    nsfw_count += 1
                # we check against nsfw<=0 to apply censorship on the explicit regions, if that happen more often (-2),
                # the prompt refiner is used in advance to avoid nsfw generation (trigger value is NSFW_WARNINGS)
                if not nsfw_check.is_safe and nsfw_check.category == NSFWCategory.EXPLICIT \
                        and (session_state.nsfw <= 0 or self.config.feature_allow_nsfw is False):
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
            if show_nsfw_censor_warning and self.config.feature_allow_nsfw:
                gr.Info("""We censored at least one of your images.
                        You can remove the censorship by uploading images to train our System for better results, or by sharing Links of this app.
                        Thanks for your understanding""", duration=0)

            try:
                # analytics
                self.analytics.record_image_creation(
                    count=len(generated_images),
                    nsfw_count=nsfw_count,
                    model=self.selectedmodelconfig.model
                )
            except Exception as e:
                logger.warning(f"error while recording sucessful image generation for stats: {e}")
                pass

        except Exception as e:
            logger.warning(f"Error while NSFW check: {e}")
            self.analytics.record_application_error(module="NSFW check", criticality="warning")
            result_images = generated_images
        return result_images

    def _apply_prompt_magic(self, session_state: SessionState, prompt: str, user_activated_promptmagic: bool) -> str:
        # check if nsfw or preview is allowed, enforce SFW prompt if not
        nsfw_preview_expired = session_state.nsfw < self.MAX_NSFW_WARNINGS
        if (not self.config.feature_allow_nsfw or nsfw_preview_expired) and self.prompt_refiner:
            nsfw, _ = self.prompt_refiner.check_contains_nsfw(prompt)
            if nsfw:
                if self.config.feature_allow_nsfw and self.config.feature_upload_images_for_new_token_enabled and \
                        not session_state.nsfw <= self.MAX_NSFW_WARNINGS * 2:
                    # and not session_state.nsfw <= self.MAX_NSFW_WARNINGS * 2: means shows warning only limited amout of time
                    gr.Info("""Your 'Preview' for explicit image generation is over and explicit content creation will now
                            be blocked by adapting your prompt.
                            You can get credits for uncensored images by uploading related images for our model training.
                            Or by sharing the Application Link. What you upload, you can create!""", duration=60)

                logger.info(f"Convert NSFW prompt to SFW. Original User-Prompt: '{prompt}'")
                prompt = self.prompt_refiner.make_prompt_sfw(prompt, True)
            else:
                logger.debug("prompt is SFW")

        if self.prompt_refiner and user_activated_promptmagic:
            logger.debug("Apply Prompt-Magic")
            # refine prompt multiple times for better reults
            userprompt = prompt
            for _ in range(random.randrange(3)):
                new_prompt = self.prompt_refiner.magic_enhance(prompt, 200)
                if len(new_prompt) > len(prompt) or prompt == userprompt: prompt = new_prompt
            # finally check that we not created nsfw by llm mistakes
            if session_state.nsfw < self.MAX_NSFW_WARNINGS and not self.prompt_refiner.is_safe_for_work(prompt):
                prompt = self.prompt_refiner.make_prompt_sfw(prompt)

        return prompt

    def _save_output_for_debug(self, gen_data: dict, userprompt: str, generated_images: list, result_images: list):
        try:
            # check saving output for validation of generation (Debug & Beta Only!!)
            if self.config.save_generated_output:
                logger.debug(f"saving images to {self.config.output_directory}")
                gen_data["userprompt"] = userprompt
                gen_data["model"] = self.selectedmodelconfig.model
                outdir = os.path.join(self.config.output_directory, get_date_subfolder(), "generation")
                for image in generated_images:
                    save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True, generation_details=gen_data)

                for image in result_images:
                    if image not in generated_images:
                        save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True,
                                                  generation_details=gen_data, appendix="-censored")
        except Exception as e:
            logger.warning(f"error while saving images: {e}")

    def _get_image_dimensions(self, aspect_ratio):
        width, height = 512, 512
        # define fallback (first element)
        for supported_ratio in self.selectedmodelconfig.aspect_ratio.values():
            width, height = ModelConfig.split_aspect_ratio(supported_ratio)

        # try to determine teh corect aspect ratio
        for supported_ratio in self.selectedmodelconfig.aspect_ratio.keys():
            if aspect_ratio.lower() in supported_ratio.lower():
                ratio = self.selectedmodelconfig.aspect_ratio[supported_ratio]
                width, height = ModelConfig.split_aspect_ratio(ratio)
                break
        return width, height
