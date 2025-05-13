from datetime import datetime

import os
import gradio as gr
import logging
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.analytics import Analytics
from .session_manager import SessionManager

import json
import shutil


# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class LinkSharingHandler:
    """
    Use to handle the UI and logic of link references
    """

    def __init__(self, session_manager: SessionManager, config: AppConfig, analytics: Analytics):
        self.config = config
        self.session_manager = session_manager
        self.analytics = analytics
        self._param_name = "r"
        self._initialize_database()

        logger.info("initialized")

    def _load_ui_dependencies(self):
        try:
            self.msg_share_links = ""
            p = "./msgs/share_links.md"
            if os.path.exists(p):
                with open(p, "r") as f:
                    self.msg_share_links = f.read()
            logger.debug(f"Initialized link msg from '{p}'")
        except Exception as e:
            logger.error(f"Error while loading link msg: {e}")

    def _initialize_database(self):
        try:
            self._session_references_database = {}  # key=referencID, value = int count(image created via reference)
            # Think about to make DB persistent, but as sessions are not persistent that could be useles
            # logger.info(f"Initialized _session_references_database from '{self.__created_images_db_path}'")
        except Exception as e:
            logger.error(f"Error while loading _session_references_database.json: {e}")

    def get_reference_code(self, request: gr.Request):
        shared_reference_key = request.query_params.get(self._param_name)
        return shared_reference_key

    def initialize_session_references(self, request: gr.Request):
        try:
            shared_reference_key = self.get_reference_code(request)
            if shared_reference_key is not None and shared_reference_key != "":
                # url = request.url
                reference_counts = self._session_references_database.get(shared_reference_key, 0)
                reference_counts += 1  # 1 point for a new session
                self._session_references_database[shared_reference_key] = reference_counts
                logger.debug(f"session reference saved for reference: {shared_reference_key}")

            self.analytics.record_new_session(
                user_agent=request.headers["user-agent"],
                languages=request.headers["accept-language"],
                reference=shared_reference_key)
        except Exception as e:
            logger.debug(f"Error while extracting reference keyfor new session: {e}")

    def record_image_generation_for_shared_link(self, request: gr.Request, image_count: int):
        try:
            shared_reference_key = self.get_reference_code(request)
            if shared_reference_key is not None and shared_reference_key != "":
                reference_counts = self._session_references_database.get(shared_reference_key, 0)
                reference_counts += image_count * self.config.feature_sharing_links_new_token_per_image
                self._session_references_database[shared_reference_key] = reference_counts
                logger.debug(f"session reference saved for reference: {shared_reference_key}")
            return shared_reference_key
        except Exception as e:
            logger.debug(f"Error while extracting reference keyfor new session: {e}")

    def earn_link_rewards(self, session_state: SessionState) -> SessionState:
        """receive all token created by link references"""
        reference_token_count = 0
        try:
            if session_state.has_reference_code() and self.config.feature_sharing_links_enabled:
                reference_token_count = self._session_references_database.get(session_state.reference_code, 0)
                if reference_token_count > 0:
                    session_state.token += reference_token_count
                    # also spend nsfw token
                    session_state.nsfw += reference_token_count // 2
                    # reset reference codes as he get alreday token
                    # FIXME: potentially not thread safe, needs to be checked
                    self._session_references_database[session_state.reference_code] = 0
                    logger.info(f"session {session_state.session} received {reference_token_count} new credits for references")
        except Exception as e:
            logger.warning(f"Reference count handling failed: {e}")
        finally:
            return session_state, reference_token_count

    def create_interface_elements(self, user_session_storage):
        if not self.config.feature_sharing_links_enabled: return
        self._load_ui_dependencies()
        # now start with interface
        with gr.Row(visible=(self.config.feature_sharing_links_enabled)):
            nsfw_msg = "(including uncensored credits)" if self.config.feature_use_upload_for_age_check else ""
            with gr.Accordion(f"Get more generator credits by sharing Links {nsfw_msg}", open=False):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(self.msg_share_links)
                    with gr.Column(scale=1):
                        create_refernce_link = gr.Button("Create Link")
                        reference_code = gr.Text(label="Share following link:", interactive=False)

                create_refernce_link.click(
                    fn=self._handle_link_creation,
                    inputs=[user_session_storage],
                    outputs=[user_session_storage, reference_code],
                    concurrency_limit=None,
                )

    def _handle_link_creation(self, request: gr.Request, gradio_state: str):
        """
        Handle token generation for image upload
        """
        if request is None:
            logger.error("no request object available")
            gr.Warning("There was a problem while generating your reference code. Please try again.")

        session_state = SessionState.from_gradio_state(gradio_state)
        try:
            logger.info(f"create reference link for session {session_state.session}")
            reference = f"{request.base_url}?{self._param_name}={session_state.get_reference_code()}"
        except Exception as e:
            logger.error(f"create reference link for session failed: {e}")
            logger.debug("Exception details:", exc_info=True)
        return session_state, reference
