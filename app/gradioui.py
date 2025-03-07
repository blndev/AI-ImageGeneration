from datetime import datetime
import io
import os
import gradio as gr
from typing import List
from PIL import Image
import logging
from app import SessionState
from app.utils.singleton import singleton
from app.utils.fileIO import save_image_with_timestamp
from app.fluxparams import FluxParameters
from app.fluxgenerator import FluxGenerator

# Set up module logger
logger = logging.getLogger(__name__)
#


@singleton
class GradioUI():
    def __init__(self):
        try:
            self.interface = None
            self.generator = FluxGenerator()
            self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)

            try:
                ars = os.getenv("ASPECT_RATIO_SQUARE", "512x512")
                self.aspect_square_width = int(ars.split("x")[0])
                self.aspect_square_height = int(ars.split("x")[1])

                arl = os.getenv("ASPECT_RATIO_LANDSCAPE", "768x512")
                self.aspect_landscape_width = int(arl.split("x")[0])
                self.aspect_landscape_height = int(arl.split("x")[1])

                arp = os.getenv("ASPECT_RATIO_PORTRAIT", "512x768")
                self.aspect_portrait_width = int(arp.split("x")[0])
                self.aspect_portrait_height = int(arp.split("x")[1])
            except Exception as e:
                logger.error("Error initializing aspect ratios: %s", str(e))
                logger.debug("Exception details:", exc_info=True)
                raise e

            self.generation_steps = int(
                os.getenv("GENERATION_STEPS", 30 if self.generator.is_flux_dev() or self.generator.SDXL else 4))
            self.generation_guidance_scale = float(
                os.getenv("GENERATION_GUIDANCE", 7.5 if self.generator.is_flux_dev() or self.generator.SDXL else 0))

            self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
            self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))
            logger.info("Initial token: %i, wait time: %i minutes", self.initial_token, self.new_token_wait_time)
            self.token_enabled = self.initial_token > 0
        except Exception as e:
            logger.error("Error initializing GradioUI: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
            raise e

    def action_session_initialized(self, request: gr.Request, session_state: SessionState):
        """Initialize analytics session when app loads.

        Args:
            request (gr.Request): The Gradio request object containing client information
            state_dict (dict): The current application state dictionary
        """
        logger.info("Session - %s - initialized with %i token for: %s",
                    session_state.session, session_state.token, request.client.host)

        # preparation for ?share=CODE
        # TODO: add to session state and to reference list shared[share_code]+=1
        # then the originated user can receive the token
        # every 600 seconds 1 token = 10 images per hour if activated
        share_value = request.query_params.get("share")
        url = request.url
        logger.debug(f"Request URL: {url}")
        # try:
        # analytics.save_session(
        #     session=session_state.session,
        #     ip=request.client.host,
        #     user_agent=request.headers["user-agent"],
        #     languages=request.headers["accept-language"])
        # logger.debug("Session - %s - saved for analytics",session_state.session)
        # except Exception as e:
        #     logger.error("Error initializing analytics session: %s", str(e))
        #     logger.debug("Exception details:", exc_info=True)

    def uiaction_timer_check_token(self, gradio_state: str):
        logger.debug(f"timer tick: {gradio_state}")
        if gradio_state is None:
            return None
        session_state = SessionState.from_gradio_state(gradio_state)
        # logic: (10) minutes after running out of token, user get filled up to initial (10) new token
        # exception: user upload image for training or receive advertising token
        if self.token_enabled:
            current_token = session_state.token
            if session_state.generation_before_minutes(self.new_token_wait_time) and session_state.token <= 0:
                session_state.token = self.initial_token
                session_state.reset_last_generation_activity()
            new_token = session_state.token - current_token
            if new_token > 0:
                gr.Info(f"Congratulation, you received {new_token} new generation token!", duration=0)
                logger.info(f"session {session_state.session} received {new_token} new token")

        return session_state

    def uiaction_generate_images(self, gradio_state, prompt, aspect_ratio, neg_prompt, image_count):
        """
        Generate images based on the given prompt and aspect ratio.

        Args:
            prompt (str): The text prompt for image generation
            aspect_ratio (str): The aspect ratio for the generated image
            image_count (int): The number of images to generate
        """
        session_state = SessionState.from_gradio_state(gradio_state)
        try:
            if self.token_enabled and session_state.token < image_count:
                raise Exception(
                    "Not enough generation token available. Please wait for a few minutes, or get more token by sharing the app.")

            session_state.save_last_generation_activity()

            # Map aspect ratio selection to dimensions
            width, height = self.aspect_square_width, self.aspect_square_height

            if "landscape" in aspect_ratio.lower():  # == "▯ Landscape (16:9)"
                width, height = self.aspect_landscape_width, self.aspect_landscape_height
            elif "portrait" in aspect_ratio.lower():  # == "▤ Portrait (2:3)"
                width, height = self.aspect_portrait_width, self.aspect_portrait_height

            logger.info(f"generating image with prompt: {prompt} and aspect ratio: {width}x{height}")

            generation_details = FluxParameters(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=self.generation_steps,
                guidance_scale=self.generation_guidance_scale,
                num_images_per_prompt=image_count,
                width=width,
                height=height
            )

            images = self.generator.generate_images(params=generation_details)
            session_state.token -= image_count
            logger.debug(f"received {len(images)} image(s) from generator")
            if self.output_directory is not None:
                logger.debug(f"saving images to {self.output_directory}")
                for image in images:
                    outdir = os.path.join(self.output_directory, datetime.now().strftime("%Y-%m-%d"))
                    save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True)
            return images, session_state
        except Exception as e:
            logger.error(f"image generation failed: {e}")
            gr.Warning(f"Error while generating the image: {e}", duration=30)
        return None, session_state

    def create_interface(self):

        # Create the interface components
        with gr.Blocks(
            title="Image Generator",
            css="footer {visibility: hidden}",
            analytics_enabled=False
        ) as self.interface:
            user_session_storage = gr.BrowserState()  # do not initialize it with any value!!! this is used as default

            # Input Values & Generate row
            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt = gr.Textbox(
                        label="Your prompt",
                        placeholder="Describe the image you want to generate...",
                        value="",
                        scale=4,
                        lines=4
                    )

                    # Text prompt input
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="Describe what you don't want...",
                        value="",
                        scale=4,
                        lines=4
                    )

                with gr.Column():
                    # Aspect ratio selection
                    aspect_ratio = gr.Radio(
                        choices=["□ Square", "▤ Landscape", "▯ Portrait"],
                        value="□ Square",
                        label="Aspect Ratio",
                        scale=1
                    )
                    image_count = gr.Slider(
                        label="Image Count",
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        scale=1
                    )

                with gr.Column(visible=True):
                    # token count is restored from app.load
                    token_label = gr.Text(
                        show_label=False,
                        value=f"?",
                        info=f"Amount of images you can generate before a wait time of {self.new_token_wait_time} minutes",
                        visible=self.token_enabled
                        )

                    # Generate button that's interactive only when prompt has text
                    generate_btn = gr.Button(
                        "Generate",
                        interactive=False
                    )
                    cancel_btn = gr.Button("Cancel", interactive=False, visible=False)

            # Examples row
            with gr.Row():
                with gr.Accordion("Examples", open=False):
                    # Examples
                    gr.Examples(
                        examples=[
                            [
                                "A majestic mountain landscape at sunset with snow-capped peaks",
                                "painting",
                                "▤ Landscape",
                                1
                            ],
                            [
                                "A professional portrait of a business person in a modern office",
                                "ugly face",
                                "▯ Portrait",
                                1
                            ],
                            [
                                "A top-down view of a colorful mandala pattern",
                                "realistic",
                                "□ Square",
                                1
                            ]
                        ],
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

            # SessionState, do not initialize it with any value!!! this is used as default
            user_session_storage = gr.BrowserState()

            def update_token_info(gradio_state):
                logger.debug("local_storage changed: SessionState: %s", gradio_state)
                token = SessionState.from_gradio_state(gradio_state).token
                if self.token_enabled == False:
                    token = "unlimited"
                return f"Generations left: {token}"
            user_session_storage.change(
                # update the UI with current token count
                fn=update_token_info,
                inputs=[user_session_storage],
                outputs=[token_label],
                concurrency_limit=None,
                show_api=False,
                show_progress=False
            )

            timer_check_token = gr.Timer(10)
            timer_check_token.tick(
                fn=self.uiaction_timer_check_token,
                inputs=[user_session_storage],
                outputs=[user_session_storage],
                concurrency_id="check_token",
                concurrency_limit=30
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
                inputs=[user_session_storage, prompt, aspect_ratio, neg_prompt, image_count],
                outputs=[gallery, user_session_storage],
                concurrency_id="gpu",
                show_progress="full"
            ).then(
                fn=lambda: (gr.Timer(active=True), gr.Button(interactive=True),
                            gr.Button(interactive=False), gr.Gallery(preview=False)),
                inputs=[],
                outputs=[timer_check_token, generate_btn, cancel_btn, gallery]
            )

            # generate_event = generate_btn.click(
            #     fn=uiaction_generate_images,
            #     inputs=[prompt, aspect_ratio, neg_prompt, image_count],
            #     outputs=[gallery]
            # )
            # Connect the cancel button

            stop_signal = gr.State(False)

            def cancel_generation():
                gr.close_all()
                return True, gr.Button(interactive=True), gr.Button(interactive=False)

            cancel_btn.click(
                fn=cancel_generation,
                inputs=[],
                outputs=[stop_signal, generate_btn, cancel_btn]
            )

            # stop_btn.click(
            #     fn=lambda: (gr.Info("Cancel")),
            #     inputs=None,
            #     outputs=None,
            #     cancels=[generate_event1, generate_event2],
            #     #queue=False
            # )

            def prepare_download(selection: gr.SelectData):
                # gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
                # Create a custom filename (for example, using timestamp)
                file_path = selection.value['image']['path']
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # custom_filename = f"generated_image_{timestamp}.png"
                return gr.DownloadButton(label=f"Download", value=file_path, visible=True)

            gallery.select(
                fn=prepare_download,
                inputs=None,
                outputs=[download_btn]
            )

            # TODO: link via fn directly to self.action...
            app = self.interface

            @app.load(inputs=[user_session_storage], outputs=[user_session_storage])
            def load_from_local_storage(request: gr.Request, gradio_state):
                # Restore token from local storage
                session_state = SessionState.from_gradio_state(gradio_state)
                logger.debug("Restoring session from local storage: %s", session_state.session)

                if gradio_state is None:
                    session_state.token = self.initial_token

                # Initialize session when app loads
                self.action_session_initialized(request=request, session_state=session_state)

                logger.debug("Restoring token from local storage: %s", session_state.token)
                logger.debug("Session ID: %s", session_state.session)
                return session_state

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.interface.launch(**kwargs)
        # self.generator.warmup()
        # gr.Interface.from_pipeline(self.generator._cached_generation_pipeline).launch()
