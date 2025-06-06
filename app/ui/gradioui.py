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
from .components import UploadHandler, SessionManager, LinkSharingHandler, ImageGenerationHandler, FeedbackHandler, PromptAssistantHandler

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
            self.analytics.register_model(selectedmodel)
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

            self.component_feedback_handler = FeedbackHandler(
                session_manager=self.component_session_manager,
                config=self.config,
                analytics=self.analytics
            )

            self.component_prompt_assistant_handler = PromptAssistantHandler(
                analytics=self.analytics,
                config=self.config,
                image_generator=self.component_image_generator
            )

            # used to determine when to unload models etc
            self.app_last_image_generation = datetime.now()

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
        # logger.debug("tick - cleanup interval")
        try:
            timeout_minutes = self.config.free_memory_after_minutes_inactivity
            x15_minutes_ago = datetime.now() - timedelta(minutes=timeout_minutes)
            self.component_session_manager.session_cleanup_and_analytics()

            if self.app_last_image_generation < x15_minutes_ago and self.component_image_generator.generator._cached_generation_pipeline:
                # no active user for x minutes, we can unload the model to free memory
                logger.debug(f"No active user for {timeout_minutes} minutes. Unloading Generator Models from Memory")
                self.component_image_generator.generator.unload_model()
        except Exception as e:
            logger.warning(f"Error in cleanup handler: {e}")
            self.analytics.record_application_error(module="cleanup", criticality="warning")

    def action_session_initialized(self, request: gr.Request, session_state: SessionState):
        """Initialize analytics session when app loads.

        Args:
            request (gr.Request): The Gradio request object containing client information
            state_dict (dict): The current application state dictionary
        """
        logger.info("Session - %s - initialized with %i (%i) credits for: %s",
                    session_state.session, session_state.token, session_state.nsfw, request.client.host)

        reference_code = ""
        if self.component_link_sharing_handler:
            reference_code = self.component_link_sharing_handler.initialize_session_references(request=request)
        if self.component_session_manager:
            self.component_session_manager.record_active_session(session_state)

        self.analytics.record_new_session(
            user_agent=request.headers.get("user-agent", "Mozilla/5.0 (compatible; MSIE 7.0; Windows NT 6.0; Win64; x64 Trident/4.0)"),
            languages=request.headers.get("accept-language", "en"),
            reference=reference_code)

    def uiaction_timer_check_token(self, gradio_state: str):
        if gradio_state is None:
            return None
        session_state = SessionState.from_gradio_state(gradio_state)
        logger.debug(f"check new credits for '{session_state.session}'. Last Generation: {session_state.last_generation}")

        session_state, new_timer_token = self.component_session_manager.check_new_token_after_wait_time(session_state)

        new_reference_token = 0
        if self.config.feature_sharing_links_enabled:
            session_state, new_reference_token = self.component_link_sharing_handler.earn_link_rewards(session_state=session_state)

        if self.config.feature_generation_credits_enabled and (new_timer_token + new_reference_token) > 0:
            msgTimer = f"{new_timer_token} for waiting," if new_timer_token > 0 else ""
            msgReference = f"{new_reference_token} for sharing links" if new_reference_token > 0 else ""
            gr.Info(f"Congratulation, you received new generation credits: {msgTimer} {msgReference}!", duration=0)

        self.analytics.update_user_tokens(session_state.session, session_state.token)
        return session_state

    def uiaction_generate_images(self, request: gr.Request, gr_state, prompt, aspect_ratio, neg_prompt, image_count, promptmagic_active, progress=gr.Progress()):
        """
        Generate images based on the given prompt and aspect ratio.

        Args:
            prompt (str): The text prompt for image generation
            aspect_ratio (str): The aspect ratio for the generated image
            image_count (int): The number of images to generate
        """
        session_state = SessionState.from_gradio_state(gr_state)
        try:
            # Record session activity
            progress(0, desc="prepare generation")
            try:
                self.component_session_manager.record_active_session(session_state)
                self.app_last_image_generation = datetime.now()
            except Exception as e:
                logger.error("Failed to record session activity: %s", str(e))
                # Continue execution as this is not critical

            session_state.save_last_generation_activity()

            # reduct image count and inform the User
            if self.config.feature_generation_credits_enabled and session_state.token > 0 and session_state.token < image_count:
                image_count = session_state.token
                msg = "You using your last generations credits!"
                if self.config.feature_upload_images_for_new_token_enabled:
                    msg += "You can get more credits by sharing images for training. Please check the section 'Upload image'."
                if self.config.feature_sharing_links_enabled:
                    msg += "Or share the application link with your reference code to other users. More details in 'Share Links'."
                gr.Warning(msg)

            # check for end of token
            if self.config.feature_generation_credits_enabled and session_state.token < image_count:
                msg = f"Not enough generation credits available.\n\nPlease wait {self.config.new_token_wait_time} minutes"
                if self.config.feature_upload_images_for_new_token_enabled:
                    msg += ", or get new credits by sharing images for training"
                if self.config.feature_sharing_links_enabled:
                    msg += ", or share the application link to other users via the section 'Sharing' ."
                logger.info("User %s attempted generation with insufficient credits (%s needed, %s available)",
                            session_state.session, image_count, session_state.token)
                gr.Warning(msg, title="Image generation failed", duration=30)
                return [], session_state, ""

            if self.component_link_sharing_handler:
                try:
                    self.component_link_sharing_handler.record_image_generation_for_shared_link(
                        request=request,
                        image_count=image_count
                    )
                except Exception as e:
                    logger.error("Failed to record image generation for shared link: %s", str(e))
                    # Continue execution as this is not critical for image generation

            try:
                progress(0.05, desc="start generation")
                generated_images, session_state, prompt = self.component_image_generator.generate_images(
                    progress=progress,
                    session_state=session_state,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    aspect_ratio=aspect_ratio,
                    image_count=image_count,
                    user_activated_promptmagic=promptmagic_active
                )
            except Exception as e:
                logger.error("Image generation failed: %s", str(e))
                logger.debug("Image generation exception details:", exc_info=True)
                # not show warning as this is done below gr.Warning(f"Failed to generate images: {str(e)}", title="Image generation failed", duration=30)
                raise Exception("AI Pipeline Error.")

            # save image hashes to prevent upload
            if self.component_upload_handler:
                try:
                    self.component_upload_handler.block_created_images_from_upload(generated_images)
                except Exception as e:
                    logger.error("Failed to block created images from upload: %s", str(e))
                    # Continue execution as this is not critical

            if session_state.token <= 1 and self.config.feature_generation_credits_enabled:
                logger.warning(f"session {session_state.session} is out of credits ({session_state.token}) left")

            progress(1, "image generation finished")
            return generated_images, session_state, prompt
        except Exception as e:
            logger.error(f"image generation failed: {e}")
            logger.debug("Exception details:", exc_info=True)

            gr.Warning(f"Error while generating the image: {e}", title="Image generation failed", duration=30)
            return [], session_state, prompt

    def create_interface(self):

        # Create the interface components
        with gr.Blocks(
            title=self.selectedmodelconfig.description + " Generator",
            css="""
            footer {visibility: hidden}
            """,
            # .smalltext {text-align: center; font-size: 8px; !important}
            # #credits {background-color: #FFCCCB;font-size: 8px; !important}
            analytics_enabled=False
        ) as self.interface:
            user_session_storage = gr.BrowserState()  # do not initialize it with any value!!! this is used as default
            gr_assistant_prompt = gr.Textbox(value=None, visible=False)
            with gr.Row():
                gr.Label(
                    label="This App is made to test our AI System. It runs maximum of 2 days and as long as we have sponsored Hardware.",
                    value=f"Generate {self.selectedmodelconfig.description} for free!"
                )
            # Input Values & Generate row
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("Assistant"):
                        gr_assistant_create_image, gr_assistant_token_info = self.component_prompt_assistant_handler.create_interface_elements(
                            session_state=user_session_storage,
                            assistant_prompt=gr_assistant_prompt
                        )

                    # TODO: move to generate Image UI
                    with gr.TabItem("Free style") as tabText:
                        with gr.Row():

                            with gr.Column(scale=2):
                                # Text prompt input
                                gr_freestyle_prompt = gr.Textbox(
                                    label="What image you want to see?",
                                    info="Prompt",
                                    placeholder="Describe the image you want to generate...",
                                    value="",
                                    lines=3,
                                    max_length=340
                                )
                                # with gr.Column():

                            # generate and token count
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():  # visible=self.config.feature_generation_credits_enabled):
                                        token_label = gr.Text(value="", container=True, show_label=False)
                                    with gr.Row():

                                        # Generate button that's interactive only when prompt has text
                                        gr_freestyle_generate_btn = gr.Button(
                                            "Start",
                                            interactive=False,
                                            variant="primary"
                                        )

                    with gr.TabItem("Examples", visible=len(self.examples) > 0):
                        def example_selected(self):
                            # FIXME generation_tabs.selected = tabText #is not working in any variation
                            gr.Info(
                                message=f"Switch now back to the Tab '{tabText.label}' and click 'Start' to generate this image.",
                                title="Example selected"
                            )

                        # Examples
                        gr.Examples(
                            examples=self.examples,
                            fn=example_selected,
                            run_on_click=True,
                            inputs=[gr_freestyle_prompt],
                            label="Click an example to load it"
                        )

            # Advanced row
            with gr.Row():
                with gr.Accordion("Advanced generation Options", open=False):
                    with gr.Tab("Image"):
                        with gr.Row():
                            with gr.Column(visible=True, scale=1):
                                with gr.Row():
                                    image_count = gr.Slider(
                                        label="Amount of Images to create",
                                        show_reset_button=False,
                                        minimum=1,
                                        maximum=int(self.selectedmodelconfig.generation.get("max_images", 2)),
                                        value=1,
                                        step=1,
                                        scale=1
                                    )
                                with gr.Row():
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
                            with gr.Column(visible=True, scale=3):
                                gr.Markdown("more options will be added here soon.")

                    with gr.Tab("Negative Prompt"):
                        with gr.Row():
                            with gr.Column(visible=True):
                                # Text prompt input
                                gr.Markdown("Write down what our generator should not create in the images:")
                                neg_prompt = gr.Textbox(
                                    label="Negative Prompt, avoid the following Elements",
                                    placeholder="Describe what you don't want...",
                                    value="",
                                    scale=3,
                                    lines=2,
                                    max_length=340,
                                    container=False
                                )
                    with gr.Tab("Prompt Magic"):
                        with gr.Row():
                            prompt_magic_checkbox = gr.Checkbox(
                                label="Enable Prompt Magic",
                                info="Optimization",
                                container=False,
                                value=(self.config.feature_prompt_magic_enabled), visible=(self.component_image_generator.promptmagic_enabled))
                        with gr.Row():
                            magic_prompt = gr.Textbox(
                                # info="used Magic Prompt",
                                show_label=True,
                                show_copy_button=True,
                                # container=False,
                                scale=2,
                                label="Used Magic Prompt",
                                lines=2,
                                interactive=False,
                                visible=prompt_magic_checkbox.value
                            )
                            prompt_magic_checkbox.change(
                                inputs=[prompt_magic_checkbox],
                                outputs=[magic_prompt],
                                fn=lambda x: gr.Textbox(visible=x)
                            )

            with gr.Accordion(
                label=f"Get Credits to generate more {("(uncensored)" if self.config.feature_use_upload_for_age_check else "")} images",
                open=False
            ):
                with gr.Tab("Share Link",
                            visible=(self.config.feature_sharing_links_enabled)):
                    # Share Links to get credits
                    if self.component_link_sharing_handler:
                        self.component_link_sharing_handler.create_interface_elements(user_session_storage)

                with gr.Tab("Upload Images",
                            visible=(self.config.output_directory and self.config.feature_upload_images_for_new_token_enabled)):
                    # Upload to get credits
                    if self.component_upload_handler:
                        self.component_upload_handler.create_interface_elements(user_session_storage)

            # Gallery row
            with gr.Row(height=800, max_height=1200):
                # Gallery for displaying generated images
                gallery = gr.Gallery(
                    label="Generated Images",
                    elem_id="image-gallery",  # used for css
                    show_share_button=False,
                    show_download_button=True,
                    format="jpeg",
                    columns=1,
                    rows=None,  # dynamic linked to image count slider
                    height="800px",
                    object_fit="cover"
                )
                # optimize space for displayed images
                image_count.change(
                    fn=lambda v: gr.Gallery(columns=v),
                    inputs=[image_count],
                    outputs=[gallery]
                )

            # Download Button
            with gr.Row():
                download_btn = gr.DownloadButton("Download", visible=False)

            # Feedback row
            feedback_txt = None
            if self.component_feedback_handler:
                feedback_txt, _ = self.component_feedback_handler.create_interface_elements(user_session_storage)

            def update_token_info(gradio_state):
                # logger.debug("local_storage changed: SessionState: %s", gradio_state)
                ss = SessionState.from_gradio_state(gradio_state)
                token = ss.token
                if self.config.feature_generation_credits_enabled is False:
                    token = "unlimited"
                n = ""
                if self.config.feature_use_upload_for_age_check and ss.nsfw > 0:
                    n = f"(Uncensored: {ss.nsfw})"
                msg = f"Amount of images you can generate: {token} {n}"
                return msg, msg
            user_session_storage.change(
                # update the UI with current token count
                fn=update_token_info,
                inputs=[user_session_storage],
                outputs=[token_label, gr_assistant_token_info],
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
            # Make button interactive only when prompt has text
            gr_freestyle_prompt.change(
                fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                inputs=[gr_freestyle_prompt],
                outputs=[gr_freestyle_generate_btn]
            )

            # it's an invisiblöe text field used to transport teh assistant prompt
            gr_assistant_prompt.change(
                fn=lambda: (gr.Timer(active=False), gr.Button(interactive=False), gr.Button(interactive=False)),
                inputs=[],
                outputs=[timer_check_token, gr_freestyle_generate_btn, gr_assistant_create_image],
            ).then(
                fn=lambda pm, ic: self.analytics.record_prompt_usage(assistant_used=True, promptmagic_used=pm, image_count=ic),
                inputs=[prompt_magic_checkbox, image_count],
                outputs=[]
            ).then(
                fn=self.uiaction_generate_images,
                inputs=[user_session_storage, gr_assistant_prompt, aspect_ratio, neg_prompt, image_count, prompt_magic_checkbox],
                outputs=[gallery, user_session_storage, magic_prompt],
                concurrency_id="gpu",
                show_progress="full",
                show_progress_on=[token_label, gr_assistant_token_info]
            ).then(
                fn=lambda: (gr.Timer(active=True), gr.Button(interactive=True)),
                inputs=[],
                outputs=[timer_check_token, gr_assistant_create_image]
            ).then(
                # enable feedback again
                fn=lambda: (gr.Textbox(interactive=True)),
                inputs=[],
                outputs=[feedback_txt]
            )

            # Connect the generate button to the generate function and disable button and token timer while generation
            gr_freestyle_generate_btn.click(
                fn=lambda: (gr.Timer(active=False), gr.Button(interactive=False), gr.Button(interactive=False), gr.Gallery(preview=False)),
                inputs=[],
                outputs=[timer_check_token, gr_freestyle_generate_btn, gr_assistant_create_image, gallery],
            ).then(
                fn=lambda pm, ic: self.analytics.record_prompt_usage(assistant_used=False, promptmagic_used=pm, image_count=ic),
                inputs=[prompt_magic_checkbox, image_count],
                outputs=[]
            ).then(
                fn=self.uiaction_generate_images,
                inputs=[user_session_storage, gr_freestyle_prompt, aspect_ratio, neg_prompt, image_count, prompt_magic_checkbox],
                outputs=[gallery, user_session_storage, magic_prompt],
                concurrency_id="gpu",
                show_progress="full",
                show_progress_on=token_label
            ).then(
                fn=lambda: (gr.Timer(active=True), gr.Button(interactive=True),
                            gr.Button(interactive=True), gr.Gallery(preview=False)),
                inputs=[],
                outputs=[timer_check_token, gr_freestyle_generate_btn, gr_assistant_create_image, gallery]
            ).then(
                # enable feedback again
                fn=lambda: (gr.Textbox(interactive=True)),
                inputs=[],
                outputs=[feedback_txt]
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
