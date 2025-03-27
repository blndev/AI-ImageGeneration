from datetime import datetime
from hashlib import sha1
import os
import gradio as gr
from typing import List
from PIL import Image, ExifTags
import logging
from app import SessionState
from .appconfig import AppConfig
from app.utils.singleton import singleton
from app.utils.fileIO import save_image_with_timestamp, get_date_subfolder
from app.generators import FluxGenerator, GenerationParameters, ModelConfig, StabelDiffusionGenerator
from app.validators import FaceDetector, PromptRefiner
from .analytics import Analytics
import json
import shutil


# Set up module logger
logger = logging.getLogger(__name__)
#


@singleton
class GradioUI():
    def __init__(self, modelconfigs: List[ModelConfig] = None):
        try:
            self.interface = None
            self.modelconfigs = modelconfigs
            self.analytics = Analytics()
            self.config = AppConfig()
            self.existing_sessions = {}  # key=sessionid, value = timestamp of last token refresh
            self.session_references = {}  # key=referencID, value = int count(image created via reference)

            self.__feedback_file = self.config.user_feedback_filestorage
            logger.info(f"Initialize Feedback file on '{self.__feedback_file}'")

            logger.debug("Initial token: %i, wait time: %i minutes", self.config.initial_token, self.config.new_token_wait_time)

            selectedmodel = self.config.selected_model
            self.selectedmodelconfig = ModelConfig.get_config(model=selectedmodel, configs=modelconfigs)

            if self.selectedmodelconfig is None:
                logger.critical(
                    f"Could not find model {selectedmodel} specified in env 'GENERATION_MODEL' or 'default' as fallback. Stopping execution")
                exit(1)

            # Sanity checks like model.Type, model.path, ...
            if self.selectedmodelconfig.path.strip() == "" or self.selectedmodelconfig.model_type.strip() == "":
                logger.error("Configured Model and parents does not contain a path or modeltype. Stop execution.")
                exit(1)

            self.initialize_database_uploaded_images()
            self.initialize_database_created_images()
            self.initialize_upload_ui()
            self.initialize_examples()
            self.initialize_image_generator()

            self.prompt_refiner = None
            self.promptmagic_enabled = False
            if self.config.feature_prompt_magic_enabled:
                self.prompt_refiner = PromptRefiner()
                self.promptmagic_enabled = self.prompt_refiner.validate_refiner_is_ready()

            if not self.promptmagic_enabled:
                logger.warning("NSFW protection via PromptMagic is turned off")

            logger.info("Application succesful initialized")

        except Exception as e:
            logger.error("Error initializing GradioUI: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
            raise e

    def initialize_image_generator(self):
        if "flux" in self.selectedmodelconfig.model_type:
            self.generator = FluxGenerator(modelconfig=self.selectedmodelconfig)
        else:
            self.generator = StabelDiffusionGenerator(appconfig=self.config, modelconfig=self.selectedmodelconfig)

        if self.config.allow_upload:
            self.face_analyzer = FaceDetector()

    def initialize_upload_ui(self):
        try:
            self.msg_share_image = ""
            p = "./msgs/sift.md"
            if os.path.exists(p):
                with open(p, "r") as f:
                    self.msg_share_image = f.read()
            logger.info(f"Initialized messages from '{p}'")
        except Exception as e:
            logger.error(f"Error while loading msgs: {e}")

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

    def initialize_database_uploaded_images(self):
        if self.config.allow_upload:
            # Fallback
            basedir = "./output/"
            if self.config.output_directory:
                basedir = self.config.output_directory
            else:
                logger.error("Upload allowed but no output directory specified. Using fallback ./output")
            try:
                self._uploaded_images = {}
                # check maybe it's better to add the data folder as well
                self.__uploaded_images_path = os.path.join(basedir, "uploaded_images.json")
                if os.path.exists(self.__uploaded_images_path):
                    with open(self.__uploaded_images_path, "r") as f:
                        self._uploaded_images.update(json.load(f))
                logger.info(f"Initialized upload files history from '{self.__uploaded_images_path}'")
            except Exception as e:
                logger.error(f"Error while loading uploaded_images.json: {e}")

    def initialize_database_created_images(self):
        if self.config.allow_upload:
            # Fallback
            basedir = "./output/"
            if self.config.output_directory:
                basedir = self.config.output_directory
            else:
                logger.error("No output directory specified. Using fallback ./output")
            try:
                self._created_images_history = {}
                # check maybe it's better to add the data folder as well
                self.__created_images_path = os.path.join(basedir, "created_images.json")
                if os.path.exists(self.__created_images_path):
                    with open(self.__created_images_path, "r") as f:
                        self._created_images_history.update(json.load(f))
                logger.info(f"Initialized created files history from '{self.__created_images_path}'")
            except Exception as e:
                logger.error(f"Error while loading created_images.json: {e}")

    def action_session_initialized(self, request: gr.Request, session_state: SessionState):
        """Initialize analytics session when app loads.

        Args:
            request (gr.Request): The Gradio request object containing client information
            state_dict (dict): The current application state dictionary
        """
        logger.info("Session - %s - initialized with %i token for: %s",
                    session_state.session, session_state.token, request.client.host)

        self.analytics.record_new_session(user_agent=request.headers["user-agent"], languages=request.headers["accept-language"])

    def uiaction_timer_check_token(self, gradio_state: str):
        if gradio_state is None:
            return None
        session_state = SessionState.from_gradio_state(gradio_state)
        logger.debug(f"check new token for '{session_state.session}'. Last Generation: {session_state.last_generation}")
        # logic: (10) minutes after running out of token, user get filled up to initial (10) new token
        # exception: user upload image for training or receive advertising token
        if self.config.token_enabled:
            current_token = session_state.token
            if session_state.generation_before_minutes(self.config.new_token_wait_time) and session_state.token <= 2:
                session_state.token = self.config.initial_token
                session_state.reset_last_generation_activity()
            new_token = session_state.token - current_token
            if new_token > 0:
                gr.Info(f"Congratulation, you received {new_token} new generation token for waiting!", duration=0)
                logger.info(f"session {session_state.session} received {new_token} new token for waiting")

        if session_state.has_reference_code() and self.config.feature_sharing_links_enabled:
            try:
                reference_count = self.session_references.get(session_state.reference_code, 0)
                if reference_count > 0:
                    new_token = reference_count * self.config.feature_sharing_new_token
                    session_state.token += new_token
                    gr.Info(
                        f"""Congratulation, other Users used your shared link to generate images.
                        You received {new_token} new generation token!""", duration=0)
                    logger.info(f"session {session_state.session} received {new_token} new token for references")
                    # FIXME: could be not thread safe:
                    self.session_references[session_state.reference_code] = 0
            except Exception as e:
                logger.warning(f"Reference count handling failed: {e}")

        # record the session as active with timestamp in local session state dictionary
        if self.existing_sessions is not None:
            self.existing_sessions[session_state.session, datetime.now()]

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

        try:
            if self.config.token_enabled and session_state.token < image_count:
                msg = f"Not enough generation token available.\n\nPlease wait {self.config.new_token_wait_time} minutes"
                if self.config.allow_upload:
                    msg += ", or get new token by sharing images for training"
                if self.config.feature_sharing_links_enabled:
                    msg += ", or share the application link to other users"
                gr.Warning(msg, title="Image generation failed", duration=30)
                return None, session_state

            session_state.save_last_generation_activity()

            # preparation for ?share=CODE
            # TODO: add to session state and to reference list shared[share_code]+=1
            # then the originated user can receive the token
            # every 600 seconds 1 token = 10 images per hour if activated
            # shoudl be in create image
            share_value = request.query_params.get("share")
            if share_value is not None and share_value != "":
                # url = request.url
                v = self.session_references.get(share_value, 0)
                v += image_count
                self.session_references[share_value] = v
                logger.debug(f"session reference saved for reference: {share_value}")

            # Map aspect ratio selection to dimensions
            width, height = 512, 512  # fallpack
            for supported_ratio in self.selectedmodelconfig.aspect_ratio.keys():
                if supported_ratio.lower() in aspect_ratio.lower():
                    ratio = self.selectedmodelconfig.aspect_ratio[supported_ratio]
                    width, height = ModelConfig.split_aspect_ratio(ratio)
                    break

            prompt = str(prompt.strip()).replace("'", "-")
            userprompt = prompt
            neg_prompt = neg_prompt.strip()
            # make default only sfw, TODO add ehancement (upload required for NSFW)
            # TODO: feature switch in config to enable NSFW and NSFW_AFTER_UPLOAD
            if self.prompt_refiner:
                if self.prompt_refiner.check_contains_nsfw(prompt):  # TODO: and config and not enabled for user
                    logger.info(f"Convert NSFW prompt to SFW. User Prompt: '{prompt}'")
                    prompt = self.prompt_refiner.make_prompt_sfw(prompt, True)
                else:
                    logger.debug("prompt is SFW")

            # TODO: check for exclusion
            if self.prompt_refiner != None:
                neg_prompt = "nude, naked, nsfw, porn," + neg_prompt

            if self.prompt_refiner and promptmagic_active:
                logger.info(f"Apply Prompt-Magic")
                prompt = self.prompt_refiner.magic_enhance(prompt, 70)
                prompt = self.prompt_refiner.make_prompt_sfw(prompt)

            logger.info(
                f"generating image for {session_state.session} with {session_state.token} token available.\n - prompt: '{prompt}'")

            generation_details = GenerationParameters(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(self.selectedmodelconfig.generation.get("steps", 30)),  # TODO: make object out of it
                guidance_scale=float(self.selectedmodelconfig.generation.get("guidance", 1)),
                # FIXME: is that only required form i2i? strength=self.selectedmodelconfig.generation["strength"],
                num_images_per_prompt=image_count,
                width=width,
                height=height
            )

            images = self.generator.generate_images(params=generation_details)
            session_state.token -= image_count
            logger.debug(f"received {len(images)} image(s) from generator")
            if self.config.output_directory is not None:
                logger.debug(f"saving images to {self.config.output_directory}")
                gen_data = generation_details.to_dict()
                gen_data["userprompt"] = userprompt
                gen_data["model"] = self.generator.modelconfig.model
                for image in images:
                    outdir = os.path.join(self.config.output_directory, get_date_subfolder(), "generation")
                    save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True, generation_details=gen_data)

            if session_state.token <= 1:
                logger.warning(f"session {session_state.session} running out of token ({session_state.token}) left")

            return images, session_state
        except Exception as e:
            logger.error(f"image generation failed: {e}")
            # log Exceptiondetails as debug
            # logger.debug(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
            logger.debug("Exception details:", exc_info=True)

            gr.Warning(f"Error while generating the image: {e}", title="Image generation failed", duration=30)
        return None, session_state

    def uiaction_image_upload(self, gradio_state: str, image_path):
        """
        Handle image upload

        Args:
            gradio_state: the session state
            image (PIL.Image): The uploaded image
        """
        logger.debug("starting image upload handler")

        if self.config.output_directory is None:
            logger.debug("output directory not configured, skipping upload")
            return gr.Button(interactive=False)

        if image_path is None:
            return gr.Button(interactive=False)

        try:
            logger.debug(f"image type: {type(image_path)} with value {image_path}")
            image = Image.open(image_path)
            filename = os.path.basename(image_path)
            session_state = SessionState.from_gradio_state(gradio_state)
            image_sha1 = sha1(image.tobytes()).hexdigest()
            logger.info(f"UPLOAD from {session_state.session} with ID: {image_sha1}")
            if self._uploaded_images.get(image_sha1) is not None:
                logger.warning(f"Image {image_sha1} already uploaded, cancel save to disk")
                # we keep upload button true as the whole logic is behind it
                # could happen that it was not uploaded by same user and get few points
                return gr.Button(interactive=True)

            dir = self.config.output_directory
            dir = os.path.join(dir, get_date_subfolder(), "upload")
            targetpath = os.path.join(dir, image_sha1 + "_" + filename)
            # copy file from source  to outdir
            os.makedirs(dir, exist_ok=True)
            shutil.copy(image_path, targetpath)
            # fn = str(session_state.session)+"_" + image_sha1+".png"
            # input_file_path = save_image_as_png(image, dir, filename=fn)
            logger.info(f"Image saved to {targetpath}")
        except Exception as e:
            logger.error(f"save image failed: {e}")
        return gr.Button(interactive=True)

    def uiaction_image_upload_token_generation(self, gradio_state: str, image_path):
        """
        Handle token generation for image upload
        """
        if image_path is None:
            logger.error("Image received a token generation is none")
            return gradio_state, gr.Button(interactive=False), None

        try:
            logger.debug(f"image type: {type(image_path)} with value {image_path}")
            image = Image.open(image_path)
            session_state = SessionState.from_gradio_state(gradio_state)
            logger.info(f"Analyze upload to receive TOKEN from {session_state.session}")
            token = 25  # TODO: load from config
            msg = ""
            image_sha1 = sha1(image.tobytes()).hexdigest()
            uploadstate = self._uploaded_images.get(image_sha1)
            if (uploadstate):
                msg = """The image signature matches a previous submission, so the full token reward isn't possible.
                We’re awarding you 5 tokens as a thank you for your involvement."""
                token = 5
                if uploadstate.get(session_state.session):
                    msg = "You've already submitted this image, and it won't generate any tokens."
                    gr.Warning(msg, title="Upload failed")
                    return session_state, gr.Button(interactive=False), None
            else:
                self._uploaded_images[image_sha1] = {session_state.session: {"token": token, "msg": ""}}

                try:
                    img_info = image.info
                    if img_info is None:
                        logger.debug("No image info found")
                        img_info = {}
                    for key, value in img_info.items():
                        print(f"IMG.Info: {key}: {value}")

                    # check for an face and ai
                    # TODO: exif is not working, refactor to extra function and create tests
                    exif_data = image.getexif()
                    if exif_data is not None:
                        logger.debug(f"{len(exif_data)} EXIF data found")
                        if len(exif_data) > 0:
                            for key, val in exif_data.items():
                                print(f'{key}:{val}')
                                if key in ExifTags.TAGS:
                                    print(f'{ExifTags.TAGS[key]}:{val}')

                            actions_description = exif_data.get('Actions Description')
                            if actions_description:
                                print(f"Actions Description: {actions_description}")
                            else:
                                print("Actions Description nicht gefunden.")

                            gps_ifd = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
                            if gps_ifd is not None and len(gps_ifd) > 0:
                                logger.debug("Image probably contains GPS data, so no AI")
                            elif "Generator" in exif_data:
                                msg = "Image probably AI generated"
                                logger.warning(msg)
                                token = 5
                            elif exif_data[ExifTags.Base.Software] == "PIL" or exif_data[ExifTags.Base.HostComputer]:
                                msg = "Image probably generated or edited"
                                logger.warning(msg)
                                token = 5
                            elif exif_data[ExifTags.Base.Copyright] != None:
                                msg = "Image is copyright protected"
                                logger.warning(msg)
                                token = 10
                        # elif exif_data[]==:
                        #     msg = "Image is copyright protected"
                        #     logger.warning(msg)
                        #     token = 10
                    else:
                        logger.debug("No EXIF data found")
                except Exception as e:
                    logger.error(f"Error while checking image EXIF data: {e}")

                try:
                    faces, cv2 = self.face_analyzer.get_faces(image)
                    if len(faces) == 0:
                        msg = """No face detected in the image. Could happen that the face is to narrow or the resoltution is too small.
                                 Try another pictrue to get more token!"""
                        token = 5
                        logger.warning(f"No Face detected on image {image_sha1} from {session_state.session}")
                    else:
                        # prepare for auto removal of critical images
                        logger.info(f"{len(faces)} Face(s) detected on upload from {session_state.session}")
                        ages = ""
                        for face in faces:
                            if face.age:
                                ages += str(face.age) + ","
                                if face.age < 18 and self.config.output_directory != None:
                                    fn = os.path.join(self.config.output_directory, "warning", get_date_subfolder())
                                    fn = os.path.join(fn, f"{image_sha1}-{face.age}.jpg")
                                    self.face_analyzer.get_face_picture(cv2, face, filename=fn)
                                    logger.warning(f"Suspected age detected on image {image_sha1}")
                        logger.debug(f"Ages on the image {image_sha1}: {ages[:-1]}")

                except Exception as e:
                    logger.error(f"Error while detecting face: {e}")

            if (token > 0):
                session_state.token += token
                if msg != "":
                    gr.Info(f"You received {token} new generation token! \n\nNotes: {msg}", duration=30)
                else:
                    gr.Info(f"Congratulation, you received {token} new generation token!", duration=30)

            else:
                gr.Warning(msg, title="Upload failed")
            # if token = 0, it was already claimed or it's failing the checks
            if not self._uploaded_images[image_sha1].get(session_state.session):
                # split creation and assignment as old data might be in the object we ant to keep
                self._uploaded_images[image_sha1][session_state.session] = {}

            self._uploaded_images[image_sha1][session_state.session]["token"] = token
            self._uploaded_images[image_sha1][session_state.session]["msg"] = msg
            self._uploaded_images[image_sha1][session_state.session]["timestamp"] = datetime.now().isoformat()
            # now save the list to disk for reuse in later sessions
            try:
                with open(self.__uploaded_images_path, "w") as f:
                    json.dump(self._uploaded_images, f, indent=4)
            except Exception as e:
                logger.error(f"Error while saving {self.__uploaded_images_path}: {e}")

        except Exception as e:
            logger.error(f"generate token for uploaded image failed: {e}")
            logger.debug("Exception details:", exc_info=True)
        return session_state, gr.Button(interactive=False), None

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
                        lines=4,
                        max_length=340
                    )

                    # Text prompt input
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
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
                        maximum=int(os.getenv("MAX_IMAGES", 2)),
                        value=1,
                        step=1,
                        scale=1
                    )
                    prompt_magic_checkbox = gr.Checkbox(
                        label="Enable Prompt Magic",
                        value=(self.prompt_refiner is not None), visible=(self.prompt_refiner is not None))
                    # gr.Markdown(label="Note",
                    #             value="""NSFW protection is active.
                    #             """, show_label=True, container=True,
                    #             visible=(self.prompt_refiner is not None))

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
                            "Generate",
                            interactive=False
                        )
                        cancel_btn = gr.Button("Cancel", interactive=False, visible=False)

            # Examples row
            with gr.Row(visible=len(self.examples) > 0):
                with gr.Accordion("Examples", open=False):
                    # Examples
                    gr.Examples(
                        examples=self.examples,
                        inputs=[prompt, neg_prompt, aspect_ratio, image_count],
                        label="Click an example to load it"
                    )

            # Upload to get Token row
            with gr.Row(visible=(self.config.output_directory and self.config.allow_upload)):
                with gr.Accordion("Get more Token", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(self.msg_share_image)
                        with gr.Column(scale=1):
                            upload_image = gr.Image(sources="upload", type="filepath", format="jpeg", height=256)
                            upload_button = gr.Button("Upload", visible=True, interactive=False)

                    upload_image.change(
                        fn=self.uiaction_image_upload,
                        inputs=[user_session_storage, upload_image],
                        outputs=[upload_button],
                        concurrency_limit=None,
                        concurrency_id="image upload"
                    )

                    upload_button.click(
                        fn=self.uiaction_image_upload_token_generation,
                        inputs=[user_session_storage, upload_image],
                        outputs=[user_session_storage, upload_button, upload_image],
                        concurrency_limit=None,
                        concurrency_id="image upload"
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
                token = SessionState.from_gradio_state(gradio_state).token
                if self.config.token_enabled is False:
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
                outputs=[gallery, user_session_storage],
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
                    session_state.token = self.config.initial_token

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
