from datetime import datetime, timedelta
from time import sleep
from typing import List
import os
import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from app import SessionState
from ..appconfig import AppConfig
from app.utils.singleton import singleton
from app.generators import ModelConfig
from ..analytics import Analytics
import json
from .components import UploadHandler, SessionManager, LinkSharingHandler, ImageGenerationHandler

# Set up module logger
logger = logging.getLogger(__name__)
#


@singleton
class GradioUI():
    def __init__(self, modelconfigs: List[ModelConfig] = None):
        try:
            self.interface = None
            self.modelconfigs = modelconfigs
            self.config = AppConfig()

            # TODO: move to AppStart and just hand over the selected model
            selectedmodel = self.config.selected_model
            self.selectedmodelconfig = ModelConfig.get_config(model=selectedmodel, configs=modelconfigs)

            if self.selectedmodelconfig is None:
                logger.critical(
                    f"Could not find model {selectedmodel} specified in env 'GENERATION_MODEL' or 'default' as fallback. Stopping execution")
                exit(1)

            # Sanity checks like model.Type, model.path, ...
            if self.selectedmodelconfig.sanity_check() == False:
                logger.error("Configured Model and parents does not contain a path or modeltype. Stop execution.")
                exit(1)

            self.analytics = Analytics()
            self.component_session_manager = SessionManager(config=self.config, analytics=self.analytics)

            self.component_image_generator = ImageGenerationHandler(
                session_manager=self.component_session_manager,
                config=self.config,
                analytics=self.analytics,
                modelconfig=self.selectedmodelconfig
            )

            self.component_upload_handler = None
            if self.config.feature_upload_images_for_new_token_enabled or self.config.feature_use_upload_for_age_check:
                self.component_upload_handler = UploadHandler(
                    session_manager=self.component_session_manager,
                    config=self.config,
                    analytics=self.analytics
                )

            self.component_link_sharing_handler = None
            if self.config.feature_sharing_links_enabled:
                self.component_link_sharing_handler = LinkSharingHandler(
                    session_manager=self.component_session_manager,
                    config=self.config,
                    analytics=self.analytics
                )

            # used to determine when to unload models etc
            self.app_last_image_generation = datetime.now()

            # TODO: MOve to feedback handler
            self.__feedback_file = self.config.user_feedback_filestorage
            logger.info(f"Initialize Feedback file on '{self.__feedback_file}'")

            self.initialize_examples()

            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.interval_cleanup_and_analytics, 'interval', max_instances=1, minutes=1, id="memory management")

            logger.info("Application succesful initialized")

        except Exception as e:
            logger.error("Error initializing GradioUI: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
            raise e

    def __del__(self):
        logger.info("cleanup ressources")
        if hasattr(self, "scheduler"):
            if self.scheduler:
                self.scheduler.shutdown(wait=False)

    def initialize_examples(self):
        self.examples = []
        try:
            if len(self.selectedmodelconfig.examples) > 0:
                self.examples = self.selectedmodelconfig.examples
                logger.info("Initialized examples from modelconfig")
            else:
                p = "./msgs/examples.json"
                if os.path.exists(p):
                    with open(p, "r") as f:
                        self.examples = json.load(f)
                logger.info(f"Initialized examples from '{p}'")
        except Exception as e:
            logger.error(f"Error while loading examples: {e}")
            self.examples = [
                [
                    "A majestic mountain landscape at sunset with snow-capped peaks",
                    "painting",
                    "\u25a4 Landscape",
                    1
                ]
            ]

    def interval_cleanup_and_analytics(self):
        """is called every 60 secdonds and:
        * updates monitoring information
        * refreshes configuration
        * unloading unused models
        """
        logger.debug("tick - cleanup interval")
        timeout_minutes = self.config.free_memory_after_minutes_inactivity
        x15_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)
        self.component_session_manager.session_cleanup_and_analytics()

        if self.app_last_image_generation < x15_minutes_ago:
            # no active user for 15 minutes, we can unload the model to free memory
            logger.info(f"No active user for {timeout_minutes} minutes. Unloading Generator Models from Memory")
            self.component_image_generator.generator.unload_model()

    def action_session_initialized(self, request: gr.Request, session_state: SessionState):
        """Initialize analytics session when app loads.

        Args:
            request (gr.Request): The Gradio request object containing client information
            state_dict (dict): The current application state dictionary
        """
        logger.info("Session - %s - initialized with %i (%i) credits for: %s",
                    session_state.session, session_state.token, session_state.nsfw, request.client.host)

        self.component_link_sharing_handler.initialize_session_references(request=request)
        self.component_session_manager.record_active_session(session_state)

    def uiaction_timer_check_token(self, gradio_state: str):
        if gradio_state is None:
            return None
        session_state = SessionState.from_gradio_state(gradio_state)
        logger.debug(f"check new credits for '{session_state.session}'. Last Generation: {session_state.last_generation}")

        session_state, new_timer_token = self.component_session_manager.check_new_token_after_wait_time(session_state)

        new_reference_token = 0
        if self.config.feature_sharing_links_enabled:
            session_state, new_reference_token = self.component_link_sharing_handler.earn_link_rewards(session_state=session_state)

        if self.config.token_enabled and (new_timer_token + new_reference_token) > 0:
            msgTimer = f"{new_timer_token} for waiting," if new_timer_token > 0 else ""
            msgReference = f"{new_reference_token} for sharing links" if new_reference_token > 0 else ""
            gr.Info(f"Congratulation, you received new generation credits: {msgTimer} {msgReference}!", duration=0)

        self.analytics.update_user_tokens(session_state.session, session_state.token)
        return session_state

    def uiaction_generate_images(self, request: gr.Request, gr_state, prompt, aspect_ratio, neg_prompt, image_count, promptmagic_active):
        """
        Generate images based on the given prompt and aspect ratio.

        Args:
            prompt (str): The text prompt for image generation
            aspect_ratio (str): The aspect ratio for the generated image
            image_count (int): The number of images to generate
        """
        session_state = SessionState.from_gradio_state(gr_state)
        self.component_session_manager.record_active_session(session_state)
        self.app_last_image_generation = datetime.now()
        analytics_image_creation_duration_start_time = None
        try:
            if self.config.token_enabled and session_state.token < image_count:
                msg = f"Not enough generation credits available.\n\nPlease wait {self.config.new_token_wait_time} minutes"
                if self.config.feature_upload_images_for_new_token_enabled:
                    msg += ", or get new credits by sharing images for training"
                if self.config.feature_sharing_links_enabled:
                    msg += ", or share the application link to other users"
                gr.Warning(msg, title="Image generation failed", duration=30)
                return [], session_state

            analytics_image_creation_duration_start_time = self.analytics.start_image_creation_timer()
            session_state.save_last_generation_activity()
            shared_reference_key = None

            if self.component_link_sharing_handler:
                shared_reference_key = self.component_link_sharing_handler.record_image_generation_for_shared_link(
                    request=request,
                    image_count=image_count
                )

            generated_images, session_state, prompt = self.component_image_generator.generate_images(
                session_state=session_state,
                prompt=prompt,
                neg_prompt=neg_prompt,
                aspect_ratio=aspect_ratio,
                image_count=image_count,
                user_activated_promptmagic=promptmagic_active
            )

            # save image hashes to prevent upload
            if self.component_upload_handler:
                self.component_upload_handler.block_created_images_from_upload(generated_images)

            if session_state.token <= 1:
                logger.warning(f"session {session_state.session} running out of credits ({session_state.token}) left")

            try:
                # TODO: create useful values for "content" eg. main focus (describe the main object create in the image in one word")
                # use blib or ollama image descibe
                self.analytics.record_image_creation(
                    count=image_count,
                    model=self.selectedmodelconfig.model,
                    content="",
                    reference=shared_reference_key)
            except Exception as e:
                logger.warning(f"error while recording sucessful image generation for stats: {e}")
                pass

            return generated_images, session_state, prompt
        except Exception as e:
            logger.error(f"image generation failed: {e}")
            logger.debug("Exception details:", exc_info=True)

            gr.Warning(f"Error while generating the image: {e}", title="Image generation failed", duration=30)
            return [], session_state, prompt

        finally:
            self.analytics.stop_image_creation_timer(analytics_image_creation_duration_start_time)

    def create_interface(self):

        # Create the interface components
        with gr.Blocks(
            title=self.selectedmodelconfig.description + " Image Generator",
            css="footer {visibility: hidden}",
            analytics_enabled=False
        ) as self.interface:
            user_session_storage = gr.BrowserState()  # do not initialize it with any value!!! this is used as default

            # with gr.Row():
            #     gr.Markdown(value=self.selectedmodelconfig.description)
            # Input Values & Generate row
            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt = gr.Textbox(
                        label="Your wish",
                        placeholder="Describe the image you want to generate...",
                        value="",
                        scale=4,
                        lines=4,
                        max_length=340
                    )

                    # Text prompt input
                    neg_prompt = gr.Textbox(
                        label="Avoid",
                        placeholder="Describe what you don't want...",
                        value="",
                        scale=4,
                        lines=4,
                        max_length=340
                    )

                with gr.Column():
                    # Aspect ratio selection
                    ratios = []
                    for k in self.selectedmodelconfig.aspect_ratio.keys():
                        ratios.append(k)
                    if len(ratios) == 0:
                        ratios.append("default")
                    aspect_ratio = gr.Radio(
                        # choices=["□ Square", "▤ Landscape", "▯ Portrait"],
                        choices=ratios,
                        value=ratios[0],
                        label="Aspect Ratio",
                        scale=1
                    )
                    image_count = gr.Slider(
                        label="Image Count",
                        minimum=1,
                        maximum=int(self.selectedmodelconfig.generation.get("max_images", 2)),
                        value=1,
                        step=1,
                        scale=1
                    )
                    prompt_magic_checkbox = gr.Checkbox(
                        label="Enable Prompt Magic",
                        value=(self.config.feature_prompt_magic_enabled), visible=(self.component_image_generator.promptmagic_enabled))
                    magic_prompt = gr.Textbox(label="Magic Prompt", interactive=False, visible=prompt_magic_checkbox.value)
                    prompt_magic_checkbox.change(
                        inputs=[prompt_magic_checkbox],
                        outputs=[magic_prompt],
                        fn=lambda x: gr.Textbox(visible=x)
                    )

                # generate and token count
                with gr.Column(visible=True):
                    with gr.Row(visible=self.config.token_enabled):
                        # token count is restored from app.load
                        token_label = gr.Text(
                            show_label=False,
                            scale=2,
                            value="?",
                            info=f"Amount of images you can generate before a wait time of {self.config.new_token_wait_time} minutes",
                        )
                        button_check_token = gr.Button(value="🗘", size="sm")
                    with gr.Row():
                        # Generate button that's interactive only when prompt has text
                        generate_btn = gr.Button(
                            "Generate " + self.selectedmodelconfig.description,
                            interactive=False
                        )
                        cancel_btn = gr.Button("Cancel", interactive=False, visible=False)

            # Sharew Links to get credits
            if self.component_link_sharing_handler:
                self.component_link_sharing_handler.create_interface_elements(user_session_storage)

            # Upload to get credits
            if self.component_upload_handler:
                self.component_upload_handler.create_interface_elements(user_session_storage)

            # Examples row
            with gr.Row(visible=len(self.examples) > 0):
                with gr.Accordion("Examples", open=False):
                    # Examples
                    gr.Examples(
                        examples=self.examples,
                        inputs=[prompt, neg_prompt, aspect_ratio, image_count],
                        label="Click an example to load it"
                    )

            # Gallery row
            with gr.Row():
                # Gallery for displaying generated images
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_share_button=False,
                    show_download_button=True,
                    format="jpeg",
                    columns=4,
                    rows=1,
                    height="auto",
                    object_fit="contain",
                    preview=False
                )

            # Download Button
            with gr.Row():
                download_btn = gr.DownloadButton("Download", visible=False)

            # Feedback row
            with gr.Row(visible=(self.__feedback_file)):
                with gr.Accordion("Feedback", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ## We’d Love Your Feedback!

                            We’re excited to bring you this Image Generator App, which is still in development!
                            Your feedback is invaluable to us—please share your thoughts on the app and the images it creates so we can make it better for you.
                            Your input helps shape the future of this tool, and we truly appreciate your time and suggestions.
                            """)
                        with gr.Column(scale=1):
                            feedback_txt = gr.Textbox(label="Please share your feedback here", lines=3, max_length=300)
                            feedback_button = gr.Button("Send Feedback", visible=True, interactive=False)
                        feedback_txt.change(
                            fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                            inputs=[feedback_txt],
                            outputs=[feedback_button]
                        )

                        def send_feedback(gradio_state, text):
                            session_state = SessionState.from_gradio_state(gradio_state)
                            logger.info(f"User Feedback from {session_state.session}: {text}")
                            try:
                                with open(self.__feedback_file, "a") as f:
                                    f.write(f"{datetime.now()} - {session_state.session}\n{text}\n\n")
                            except Exception:
                                pass
                            gr.Info("""
                            Thank you so much for taking the time to share your feedback!
                            We really appreciate your input—it means a lot to us and helps us make the App better for everyone."
                            """)
                        feedback_button.click(
                            fn=send_feedback,
                            inputs=[user_session_storage, feedback_txt],
                            outputs=[],
                            concurrency_limit=None,
                            concurrency_id="feedback"
                        ).then(
                            fn=lambda: (gr.Textbox(value="", interactive=False), gr.Button(interactive=False)),
                            inputs=[],
                            outputs=[feedback_txt, feedback_button]
                        )

            def update_token_info(gradio_state):
                # logger.debug("local_storage changed: SessionState: %s", gradio_state)
                ss = SessionState.from_gradio_state(gradio_state)
                token = ss.token
                if self.config.token_enabled is False:
                    token = "unlimited"
                n = ""
                if self.config.feature_use_upload_for_age_check and ss.nsfw > 0:
                    n = f"(Uncensored: {ss.nsfw})"
                return f"Generations left: {token} {n}"
            user_session_storage.change(
                # update the UI with current token count
                fn=update_token_info,
                inputs=[user_session_storage],
                outputs=[token_label],
                concurrency_limit=None,
                show_api=False,
                show_progress=False
            )

            timer_check_token = gr.Timer(60)
            timer_check_token.tick(
                fn=self.uiaction_timer_check_token,
                inputs=[user_session_storage],
                outputs=[user_session_storage],
                concurrency_id="check_token",
                concurrency_limit=30
            )
            button_check_token.click(
                fn=self.uiaction_timer_check_token,
                inputs=[user_session_storage],
                outputs=[user_session_storage],
                concurrency_id="check_token",
            ).then(
                # enable feedback again
                fn=lambda: (sleep(2)),
                inputs=[],
                outputs=[]
            )

            # Make button interactive only when prompt has text
            prompt.change(
                fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                inputs=[prompt],
                outputs=[generate_btn]
            )

            # Connect the generate button to the generate function and disable button and token timer while generation
            generate_btn.click(
                fn=lambda: (gr.Timer(active=False), gr.Button(interactive=False), gr.Button(interactive=True)),
                inputs=[],
                outputs=[timer_check_token, generate_btn, cancel_btn],
            ).then(
                fn=self.uiaction_generate_images,
                inputs=[user_session_storage, prompt, aspect_ratio, neg_prompt, image_count, prompt_magic_checkbox],
                outputs=[gallery, user_session_storage, magic_prompt],
                concurrency_id="gpu",
                show_progress="full"
            ).then(
                fn=lambda: (gr.Timer(active=True), gr.Button(interactive=True),
                            gr.Button(interactive=False), gr.Gallery(preview=False)),
                inputs=[],
                outputs=[timer_check_token, generate_btn, cancel_btn, gallery]
            ).then(
                # enable feedback again
                fn=lambda: (gr.Textbox(interactive=True)),
                inputs=[],
                outputs=[feedback_txt]
            )

            stop_signal = gr.State(False)

            def cancel_generation():
                gr.close_all()
                return True, gr.Button(interactive=True), gr.Button(interactive=False)

            cancel_btn.click(
                fn=cancel_generation,
                inputs=[],
                outputs=[stop_signal, generate_btn, cancel_btn]
            )

            def prepare_download(selection: gr.SelectData):
                # gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
                # Create a custom filename (for example, using timestamp)
                file_path = selection.value['image']['path']
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # custom_filename = f"generated_image_{timestamp}.png"
                return gr.DownloadButton(label="Download", value=file_path, visible=True)

            gallery.select(
                fn=prepare_download,
                inputs=None,
                outputs=[download_btn]
            )

            app = self.interface

            @app.load(inputs=[user_session_storage], outputs=[user_session_storage])
            def load_from_local_storage(request: gr.Request, gradio_state):
                # Restore token from local storage
                session_state = SessionState.from_gradio_state(gradio_state)
                logger.debug("Restoring session from local storage: %s", session_state.session)

                if gradio_state is None:
                    session_state.token = self.config.initial_token

                # Initialize session when app loads
                self.action_session_initialized(request=request, session_state=session_state)

                logger.debug("Restoring credits from local storage: %s", session_state.token)
                logger.debug("Session ID: %s", session_state.session)
                return session_state

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.scheduler.start()
        self.interface.launch(**kwargs)
        # self.generator.warmup()
        # gr.Interface.from_pipeline(self.generator._cached_generation_pipeline).launch()
