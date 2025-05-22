from datetime import datetime
import os
import gradio as gr
import logging
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.analytics import Analytics
from .session_manager import SessionManager

# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class FeedbackHandler:
    """
    Handler for user feedback functionality
    """

    def __init__(self, session_manager: SessionManager, config: AppConfig, analytics: Analytics):
        self.config = config
        self.session_manager = session_manager
        self.analytics = analytics
        self.__feedback_file = self.config.user_feedback_filestorage
        logger.info(f"Initialize Feedback file on '{self.__feedback_file}'")

    def _load_ui_dependencies(self):
        try:
            self.feedback_message = """
## We'd Love Your Feedback!

We're excited to bring you this Image Generator App, which is still in development!
Your feedback is invaluable to us—please share your thoughts on the app and the images it creates so we can make it better for you.
Your input helps shape the future of this tool, and we truly appreciate your time and suggestions.
"""
            p = "./msgs/feedback.md"
            if os.path.exists(p):
                with open(p, "r") as f:
                    self.feedback_message = f.read()
            logger.debug(f"Initialized feedback message from '{p}'")
        except Exception as e:
            logger.error(f"Error while loading feedback message: {e}")

    def create_interface_elements(self, user_session_storage):
        if not self.__feedback_file:
            return
            
        self._load_ui_dependencies()
        
        with gr.Row(visible=(self.__feedback_file)):
            with gr.Accordion("Feedback", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(self.feedback_message)
                    with gr.Column(scale=1):
                        feedback_txt = gr.Textbox(label="Please share your feedback here", lines=3, max_length=300)
                        feedback_button = gr.Button("Send Feedback", visible=True, interactive=False)
                    feedback_txt.change(
                        fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                        inputs=[feedback_txt],
                        outputs=[feedback_button]
                    )

                    feedback_button.click(
                        fn=self.send_feedback,
                        inputs=[user_session_storage, feedback_txt],
                        outputs=[],
                        concurrency_limit=None,
                        concurrency_id="feedback"
                    ).then(
                        fn=lambda: (gr.Textbox(value="", interactive=False), gr.Button(interactive=False)),
                        inputs=[],
                        outputs=[feedback_txt, feedback_button]
                    )
        return [feedback_txt, feedback_button]
    
    def send_feedback(self, gradio_state, text):
        """
        Handle feedback submission
        """
        session_state = SessionState.from_gradio_state(gradio_state)
        logger.info(f"User Feedback from {session_state.session}: {text}")
        try:
            with open(self.__feedback_file, "a") as f:
                f.write(f"{datetime.now()} - {session_state.session}\n{text}\n\n")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            pass
            
        gr.Info("""
        Thank you so much for taking the time to share your feedback!
        We really appreciate your input—it means a lot to us and helps us make the App better for everyone.
        """)