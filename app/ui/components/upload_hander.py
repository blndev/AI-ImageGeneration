from datetime import datetime
from hashlib import sha1

import os
import gradio as gr
from PIL import Image
import logging
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.utils.fileIO import get_date_subfolder
from app.validators import AIImageDetector, FaceDetector, NSFWDetector, NSFWCategory
from app.analytics import Analytics
from .session_manager import SessionManager

import json
import shutil


# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class UploadHandler:
    def __init__(self, session_manager: SessionManager, config: AppConfig, analytics: Analytics):
        self.config = config
        self.session_manager = session_manager
        self.analytics = analytics

        # Fallback folder
        self.basedir = "./output/"
        if self.config.output_directory:
            self.basedir = self.config.output_directory
        else:
            logger.error("No output directory specified. Using fallback ./output")

        self._initialize_database_uploaded_images()
        self._initialize_database_created_images()

    def load_components(self):
        self.nsfw_detector = NSFWDetector(confidence_threshold=0.7)
        self.face_analyzer = FaceDetector()
        self.ai_image_detector = AIImageDetector()

        try:
            self.msg_share_image = ""
            p = "./msgs/share_images.md"
            if os.path.exists(p):
                with open(p, "r") as f:
                    self.msg_share_image = f.read()
            logger.debug(f"Initialized info for upload from '{p}'")
        except Exception as e:
            logger.error(f"Error while loading msgs: {e}")

    def _initialize_database_uploaded_images(self):
        self._uploaded_images_data = {}

        try:
            self._uploaded_images_data = {}
            # check maybe it's better to add the data folder as well
            self.__uploaded_images_db_path = os.path.join(self.config.output_directory, "uploaded_images.json")
            if os.path.exists(self.__uploaded_images_db_path):
                with open(self.__uploaded_images_db_path, "r") as f:
                    self._uploaded_images_data.update(json.load(f))
            logger.info(f"Initialized upload files history from '{self.__uploaded_images_db_path}'")
        except Exception as e:
            logger.error(f"Error while loading uploaded_images.json: {e}")

    def _initialize_database_created_images(self):
        self._created_images_data = {}

        try:
            # check maybe it's better to add the data folder as well
            self.__created_images_db_path = os.path.join(self.config.output_directory, "created_images.json")
            if os.path.exists(self.__created_images_db_path):
                with open(self.__created_images_db_path, "r") as f:
                    self._created_images_data.update(json.load(f))
            logger.info(f"Initialized created files history from '{self.__created_images_db_path}'")
        except Exception as e:
            logger.error(f"Error while loading created_images.json: {e}")

    def block_created_images_from_upload(self, images):
        try:
            for image in images:
                image_sha1 = sha1(image.tobytes()).hexdigest()
                self._created_images_data[image_sha1] = True
        except Exception as e:
            logger.debug(f"Error while blocking generated images from upload: {e}")
        # now save the list to disk for reuse in later sessions
        try:
            with open(self.__created_images_db_path, "w") as f:
                json.dump(self._created_images_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error while saving {self.__created_images_db_path}: {e}")

    def create_interface_elements(self, user_session_storage):
        if not self.config.feature_upload_images_for_new_token_enabled: return
        self.load_components()
        # now start with interface
        #with gr.Row(visible=(self.config.output_directory and self.config.feature_upload_images_for_new_token_enabled)):
        if not (self.config.output_directory and self.config.feature_upload_images_for_new_token_enabled): return
        nsfw_msg = " and remove censorship by uploading explicit images!" if self.config.feature_use_upload_for_age_check else ""
        with gr.Row():
            gr.Label("Get more image generation credits" + nsfw_msg, container=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(self.msg_share_image)
            with gr.Column(scale=1):
                upload_image = gr.Image(sources="upload", type="filepath", format="jpeg", height=256)
                upload_button = gr.Button("Upload", visible=True, interactive=False)

        upload_image.change(
            fn=self._handle_upload,
            inputs=[user_session_storage, upload_image],
            outputs=[upload_button],
            concurrency_limit=None,
            concurrency_id="image upload"
        )

        upload_button.click(
            fn=self._handle_token_generation,
            inputs=[user_session_storage, upload_image],
            outputs=[user_session_storage, upload_button, upload_image],
            concurrency_limit=None,
            concurrency_id="gpu"
        )

    def _handle_upload(self, gradio_state: str, image_path: str):
        """
        Handle image upload

        Args:
            gradio_state: the session state
            image (PIL.Image): The uploaded image
        """
        logger.debug("starting image upload handler")
        session_state = SessionState.from_gradio_state(gradio_state)
        self.session_manager.record_active_session(session_state)

        if image_path is None:
            return gr.Button(interactive=False)

        try:
            logger.debug(f"image type: {type(image_path)} with value {image_path}")
            image = Image.open(image_path)
            filename = os.path.basename(image_path)

            image_sha1 = sha1(image.tobytes()).hexdigest()
            logger.info(f"UPLOAD from {session_state.session} with ID: {image_sha1}")
            if self._uploaded_images_data.get(image_sha1) is not None:
                logger.warning(f"Image {image_sha1} already uploaded, cancel save to disk")
                # we keep upload button true as the whole logic is behind it
                # this is important to allow physical upload while generation is runnning (use wait time)
                # and do the token update after generation to avoid loose received token
                return gr.Button(interactive=True)

            dir = self.config.output_directory
            #dir = os.path.join(dir, get_date_subfolder(), "upload")
            dir = os.path.join(dir, "upload")
            targetpath = os.path.join(dir, str(session_state.session) + '_' + image_sha1 + "_" + filename)
            # copy file from source to outdir
            os.makedirs(dir, exist_ok=True)
            shutil.copy(image_path, targetpath)
            logger.debug(f"Image saved to {targetpath}")
        except Exception as e:
            logger.error(f"save image failed: {e}")
        return gr.Button(interactive=True)

    def _handle_token_generation(self, gradio_state: str, image_path):
        """
        Handle token generation for image upload
        """
        if image_path is None:
            logger.error("No Image received, path is none")
            return gradio_state, gr.Button(interactive=False), None

        session_state = SessionState.from_gradio_state(gradio_state)
        self.session_manager.record_active_session(session_state)
        try:
            logger.debug(f"image type: {type(image_path)} with value {image_path}")
            image = Image.open(image_path)
            logger.info(f"Analyze upload to receive credits from {session_state.session}")
            token = self.config.feature_upload_images_token_reward
            msg = ""
            image_sha1 = sha1(image.tobytes()).hexdigest()
            already_used = self._uploaded_images_data.get(image_sha1)
            if (already_used):
                msg = """The image signature matches a previous submission, so the full credit reward isn't possible.
                We’re awarding you 5 credits as a thank you for your involvement."""
                token = 5
                if already_used.get(session_state.session):
                    msg = "You've already submitted this image, and it won't generate any credits."
                    token = 0
                    # gr.Warning(msg, title="Upload failed")
                    # return session_state, gr.Button(interactive=False), None
            elif self._created_images_data.get(image_sha1):
                msg = "This image is already known, and it won't generate any credits."
                token = 0
                # gr.Warning(msg, title="Upload failed")
                # return session_state, gr.Button(interactive=False), None
            elif (image.width <= 768 and image.height <= 768) or (image.width <= 512 or image.height <= 512):
                msg = "This image is too small to be used for training of our generator. Please use a resolution with more then 768x768 to get more credits."
                token = 1
            else:
                # prepare upload state, will be adapted later
                self._uploaded_images_data[image_sha1] = {session_state.session: {"token": token, "msg": ""}}

                try:
                    faces, cv2 = self.face_analyzer.get_faces(image)

                    is_ai_image, reason = self.ai_image_detector.is_ai_image(image_path)
                    if is_ai_image:
                        msg = "Image probably AI generated"
                        logger.info(msg + " Reason: " + reason)
                        token = 1
                    elif len(faces) == 0:
                        msg = """No face detected in the image. Could happen that the face is to narrow or the resolution is too low.
                                Try another pictrue to get more credits!"""
                        token = 5
                        logger.debug(f"No Face detected on image {image_sha1} from {session_state.session}")

                    else:
                        # prepare for auto removal of critical images
                        logger.debug(f"{len(faces)} Face(s) detected on upload from {session_state.session}")
                        ages = ""
                        for face in faces:
                            if face.age:
                                ages += str(face.age) + ","
                                if face.age < 18 and self.config.output_directory is not None:
                                    fn = os.path.join(self.config.output_directory, "warning", get_date_subfolder())
                                    fn = os.path.join(fn, f"{image_sha1}-{face.age}.jpg")
                                    self.face_analyzer.get_face_picture(cv2, face, filename=fn)
                                    logger.warning(f"Suspected age detected on image {image_sha1}")
                        logger.debug(f"Ages on the image {image_sha1}: {ages[:-1]}")

                        nsfw_result = self.nsfw_detector.detect(image)
                        if not nsfw_result.is_safe:
                            # no check required if user is prooved adult
                            logger.info(f"Upload NSFW check: {nsfw_result.category}")
                            if session_state.nsfw < 0: session_state.nsfw = 0
                            nsfwtoken = token // 2
                            if nsfw_result.category == NSFWCategory.EXPLICIT: nsfwtoken = token - 2
                            if nsfw_result.category == NSFWCategory.SUGGESTIVE: nsfwtoken = 3

                            # token += nsfwtoken
                            session_state.nsfw += nsfwtoken
                            if self.config.feature_use_upload_for_age_check: msg += f"NSFW enabled for {nsfwtoken} generations."

                except Exception as e:
                    logger.error(f"Error while analyzing uploaded image: {e}")

            if (token > 0):
                session_state.token += token
                logger.info(f"Received token for upload: {token} - {msg}")
                if msg != "":
                    gr.Info(f"You received {token} new generation credits! \n\nNote: {msg}", duration=30)
                else:
                    gr.Info(f"Congratulation, you received {token} new generation credits!", duration=30)

            else:
                gr.Warning(msg, title="Upload failed")

            # if token = 0, it was already claimed or it's failing the checks
            if not self._uploaded_images_data.get(image_sha1):
                self._uploaded_images_data[image_sha1] = {}
            if not self._uploaded_images_data[image_sha1].get(session_state.session):
                # split creation and assignment as old data might be in the object we want to keep
                self._uploaded_images_data[image_sha1][session_state.session] = {}

            self._uploaded_images_data[image_sha1][session_state.session]["token"] = token
            self._uploaded_images_data[image_sha1][session_state.session]["msg"] = msg
            self._uploaded_images_data[image_sha1][session_state.session]["timestamp"] = datetime.now().isoformat()
            # now save the list to disk for reuse in later sessions
            try:
                with open(self.__uploaded_images_db_path, "w") as f:
                    json.dump(self._uploaded_images_data, f, indent=4)
            except Exception as e:
                logger.error(f"Error while saving {self.__uploaded_images_db_path}: {e}")

        except Exception as e:
            logger.error(f"generate credits for uploaded image failed: {e}")
            logger.debug("Exception details:", exc_info=True)
        return session_state, gr.Button(interactive=False), None
