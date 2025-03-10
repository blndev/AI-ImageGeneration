from datetime import datetime
from hashlib import sha1
import io
import os
import gradio as gr
from typing import List
from PIL import Image, ExifTags
import logging
from app import SessionState
from app.utils.singleton import singleton
from app.utils.fileIO import save_image_with_timestamp, save_image_as_png, get_date_subfolder
from app.generators import FluxGenerator, FluxParameters
from app.validators.FaceDetector import FaceDetector

import json
import shutil
from distutils.util import strtobool

# Set up module logger
logger = logging.getLogger(__name__)
#


@singleton
class GradioUI():
    def __init__(self):
        try:
            self.interface = None

            self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)
            if self.output_directory == None:
                self.__feedback_file = "./output/feedback.txt"
            else:
                self.__feedback_file = os.path.join(self.output_directory, get_date_subfolder(),"feedback.txt")
            
            logger.info(f"Initialize Feedback file on '{self.__feedback_file}'")
                

            self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
            self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))
            logger.info("Initial token: %i, wait time: %i minutes", self.initial_token, self.new_token_wait_time)
            self.token_enabled = self.initial_token > 0

            self.allow_upload = bool(strtobool(os.getenv("ALLOW_UPLOAD", "False")))

            if self.allow_upload:
                # Fallback
                basedir = "./output/"
                if self.output_directory != None:
                    basedir = self.output_directory
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
            try:
                self.msg_share_image = ""
                p = "./msgs/sift.md"
                if os.path.exists(p):
                    with open(p, "r") as f:
                        self.msg_share_image = f.read()
                logger.info(f"Initialized messages from '{p}'")
            except Exception as e:
                logger.error(f"Error while loading msgs: {e}")

            # loading examples
            self.examples = []
            try:
                p = "./msgs/examples.json"
                # with open(p, "w") as f:
                #         json.dump(self.examples, f, indent=4)
                if os.path.exists(p):
                    with open(p, "r") as f:
                        self.examples = json.load(f)
                logger.info(f"Initialized examples from '{p}'")
            except Exception as e:
                logger.error(f"Error while loading examples: {e}")

            self.generator = FluxGenerator()
            self.face_analyzer = FaceDetector()

            logger.info(f"reading apect ratios and generation settings ... ")

            try:
                #TODO: add unittest which checks env and internal settings
                ars = os.getenv("GENERATION_ASPECT_SQUARE", "512x512")
                self.aspect_square_width = int(ars.split("x")[0])
                self.aspect_square_height = int(ars.split("x")[1])

                arl = os.getenv("GENERATION_ASPECT_LANDSCAPE", "768x512")
                self.aspect_landscape_width = int(arl.split("x")[0])
                self.aspect_landscape_height = int(arl.split("x")[1])

                arp = os.getenv("GENERATION_ASPECT_PORTRAIT", "512x768")
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
            self.generation_strength = float(
                os.getenv("GENERATION_STRENGHT", 0.8))
            
            logger.info(f"Application succesful initialized")

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
        if gradio_state is None:
            return None
        session_state = SessionState.from_gradio_state(gradio_state)
        logger.debug(f"check new token for '{session_state.session}'. Last Generation: {session_state.last_generation}")
        # logic: (10) minutes after running out of token, user get filled up to initial (10) new token
        # exception: user upload image for training or receive advertising token
        if self.token_enabled:
            current_token = session_state.token
            if session_state.generation_before_minutes(self.new_token_wait_time) and session_state.token <= 2:
                session_state.token = self.initial_token
                session_state.reset_last_generation_activity()
            new_token = session_state.token - current_token
            if new_token > 0:
                gr.Info(f"Congratulation, you received {new_token} new generation token!", duration=0)
                logger.info(f"session {session_state.session} received {new_token} new token for waiting")

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
                msg = f"Not enough generation token available.\n\nPlease wait {self.new_token_wait_time} minutes"
                if self.allow_upload:
                    msg+=", or get new token by sharing images for training"
                gr.Warning(msg, title="Image generation failed", duration=30)
                return None, session_state
                    

            session_state.save_last_generation_activity()

            # Map aspect ratio selection to dimensions
            width, height = self.aspect_square_width, self.aspect_square_height

            if "landscape" in aspect_ratio.lower():  # == "▯ Landscape (16:9)"
                width, height = self.aspect_landscape_width, self.aspect_landscape_height
            elif "portrait" in aspect_ratio.lower():  # == "▤ Portrait (2:3)"
                width, height = self.aspect_portrait_width, self.aspect_portrait_height

            logger.info(f"generating image for {session_state.session} with {session_state.token} token available.\n - prompt: '{prompt}' \n - Infos: aspect ratio {width}x{height}")

            generation_details = FluxParameters(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=self.generation_steps,
                guidance_scale=self.generation_guidance_scale,
                strength=self.generation_strength,
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
                    outdir = os.path.join(self.output_directory, get_date_subfolder(),"generation")
                    save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True, generation_details=generation_details.to_dict())

            if session_state.token<=1:
                logger.warning(f"session {session_state.session} running out of token ({session_state.token}) left")

            return images, session_state
        except Exception as e:
            logger.error(f"image generation failed: {e}")
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

        if self.output_directory is None:
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
            
            dir = self.output_directory
            dir = os.path.join(dir, get_date_subfolder(),"upload")
            targetpath = os.path.join(dir,  image_sha1 + "_" + filename)
            #copy file from source  to outdir
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
            logger.error(f"Image received a token generation is none")
            return gradio_state, gr.Button(interactive=False), None
        
        try:
            logger.debug(f"image type: {type(image_path)} with value {image_path}")
            image = Image.open(image_path)
            session_state = SessionState.from_gradio_state(gradio_state)
            logger.info(f"Analyze upload to receive TOKEN from {session_state.session}")
            token = 25  #TODO: load from config
            msg = ""
            image_sha1 = sha1(image.tobytes()).hexdigest()
            uploadstate = self._uploaded_images.get(image_sha1)
            if(uploadstate):
                msg = "The image signature matches a previous submission, so the full token reward isn't possible. We’re awarding you 5 tokens as a thank you for your involvement."
                token = 5
                if uploadstate.get(session_state.session) != None:
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
                    if not exif_data is None:
                        logger.debug(f"{len(exif_data)} EXIF data found")
                        if len(exif_data)>0:
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
                            if gps_ifd is not None and len(gps_ifd)>0:
                                logger.debug("Image probably contains GPS data, so no AI")
                            elif "Generator" in exif_data:
                                msg = "Image probably AI generated"
                                logger.warning(msg)
                                token = 5
                            elif exif_data[ExifTags.Base.Software] == "PIL" or exif_data[ExifTags.Base.HostComputer] != None:
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
                    if len(faces)==0:
                        msg = "No face detected in the image. Could happen that the face is to narrow or the resoltution is too small. Try another pictrue to get more token!"
                        token = 5
                        logger.warning(f"No Face detected on image {image_sha1} from {session_state.session}")
                    else:
                        # prepare for auto removal of critical images
                        logger.info(f"{len(faces)} Face(s) detected on upload from {session_state.session}")
                        ages = ""
                        for face in faces:
                            if face.age:
                                ages+=str(face.age)+","
                                if face.age<18 and self.output_directory!= None:
                                    fn = os.path.join(self.output_directory,"warning",get_date_subfolder())
                                    fn = os.path.join(fn,f"{image_sha1}-{face.age}.jpg")
                                    self.face_analyzer.get_face_picture(cv2, face, filename=fn)
                                    logger.warning(f"Suspected age detected on image {image_sha1}")
                        logger.debug(f"Ages on the image {image_sha1}: {ages[:-1]}")


                except Exception as e:
                    logger.error(f"Error while detecting face: {e}")
        
            if (token>0):
                session_state.token += token
                if msg != "":
                    gr.Info(f"You received {token} new generation token! \n\nNotes: {msg}", duration=30)
                else:
                    gr.Info(f"Congratulation, you received {token} new generation token!", duration=30)
    
            else:
                gr.Warning(msg, title="Upload failed")
            #if token = 0, it was already claimed or it's failing the checks
            if not self._uploaded_images[image_sha1].get(session_state.session):
                # split creation and assignment as old data might be in the object we ant to keep
                self._uploaded_images[image_sha1][session_state.session] = {}

            self._uploaded_images[image_sha1][session_state.session]["token"] = token
            self._uploaded_images[image_sha1][session_state.session]["msg"]=msg
            self._uploaded_images[image_sha1][session_state.session]["timestamp"]=datetime.now().isoformat()
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
                    with gr.Row(visible=self.token_enabled):
                        # token count is restored from app.load
                        token_label = gr.Text(
                            show_label=False,
                            scale=2,
                            value=f"?",
                            info=f"Amount of images you can generate before a wait time of {self.new_token_wait_time} minutes",
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
            with gr.Row(visible=len(self.examples)>0):
                with gr.Accordion("Examples", open=False):
                    # Examples
                    gr.Examples(#TODO: read samples from file
                        examples=self.examples,
                        inputs=[prompt, neg_prompt, aspect_ratio, image_count],
                        label="Click an example to load it"
                    )

            # Upload to get Token row
            with gr.Row(visible=(self.output_directory!=None and self.allow_upload)):
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
            with gr.Row(visible=(self.__feedback_file!=None)):
                with gr.Accordion("Feedback", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("""
## We’d Love Your Feedback!

We’re excited to bring you this Image Generator App, which is still in development! Your feedback is invaluable to us—please share your thoughts on the app and the images it creates so we can make it even better for you. Your input helps shape the future of this tool, and we truly appreciate your time and suggestions.
                         """                            )
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
                            gr.Info("Thank you so much for taking the time to share your feedback! We really appreciate your input—it means a lot to us and helps us make the Image Generator App better for everyone.")
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
                #logger.debug("local_storage changed: SessionState: %s", gradio_state)
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
                inputs=[user_session_storage, prompt, aspect_ratio, neg_prompt, image_count],
                outputs=[gallery, user_session_storage],
                concurrency_id="gpu",
                show_progress="full"
            ).then(
                fn=lambda: (gr.Timer(active=True), gr.Button(interactive=True),
                            gr.Button(interactive=False), gr.Gallery(preview=False)),
                inputs=[],
                outputs=[timer_check_token, generate_btn, cancel_btn, gallery]
            ).then(
                #enable feedback again
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
