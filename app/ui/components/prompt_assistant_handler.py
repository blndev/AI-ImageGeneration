from datetime import datetime
import os
import gradio as gr
import logging
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.analytics import Analytics


# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class PromptAssistantHandler:
    """
    Handler for user feedback functionality
    """

    def __init__(self, analytics: Analytics):
        self.analytics = analytics

    def _load_ui_dependencies(self):
        try:
            self.feedback_message = ""
        except Exception as e:
            logger.error(f"Error while loading assistant dependencies: {e}")

    def toggle_people_details_visibility(self, is_checked):
        return gr.Group(visible=is_checked)

    def create_interface_elements(self):
        self._load_ui_dependencies()
            
        with gr.Row():
            gr.Markdown("This section helps you to create a prompt. Just select the elements you want to have in the image")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Choose People")
                chkMale = gr.Checkbox(label="Male", value=True)
                with gr.Group() as male_group:
                    gr.Slider(interactive=True, minimum=5, maximum=80, value=25, step=10, label="Age", info="Choose the Age of that People")
                    gr.Dropdown(
                        ["Smiling", "Neutral", "Angry", "Sad"], 
                        value="Smiling", 
                        multiselect=False, label="Facial expression",
                        interactive=True
                    )
                    gr.Dropdown(
                        ["Standing", "Sitting", "Walking", "Dynamic Pose"], 
                        value="Standing", 
                        multiselect=False, label="Pose",
                        interactive=True
                    )
                    gr.Dropdown(
                        ["Casual", "Business dress", "Shorts", "Hat"], 
                        value=["Business dress"], 
                        multiselect=True, label="Clothes",
                        interactive=True
                    )
                chkMale.change(fn=self.toggle_people_details_visibility, inputs=chkMale, outputs=male_group)
                
                chkFemale = gr.Checkbox(label="Female", value=True)
                with gr.Group(visible=True) as female_group:
                    gr.Slider(interactive=True,minimum=5, maximum=80, value=25, step=10, label="Age", info="Choose the Age of that People")
                    gr.Dropdown(
                        ["Smiling", "Neutral", "Angry", "Sad"], 
                        value="Smiling", 
                        multiselect=False, label="Facial expression",
                        interactive=True
                    )
                    gr.Dropdown(
                        ["Posing", "Sitting", "Walking", "Dynamic Pose"], 
                        value="Posing", 
                        multiselect=False, label="Pose",
                        interactive=True
                    )
                    gr.Dropdown(
                        ["Shorts", "Tank Top", "Casual", "Swimwear", "Sunglasses"], 
                        value=["Casual", "Sunglasses"], 
                        multiselect=True, label="Clothes",
                        interactive=True
                    )
                # Ensure the female group is always visible
                chkFemale.change(fn=self.toggle_people_details_visibility, inputs=chkFemale, outputs=female_group)
            with gr.Column():
                gr.Markdown("Environment")
                gr.Dropdown(
                    ["Home", "Backyard", "Forest", "Beach", "Castle", "Photo Studio"], 
                    value="Photo Studio", 
                    multiselect=False, label="Location",
                    interactive=True
                )
                with gr.Row():
                    feedback_button = gr.Button("Create Prompt", visible=True, interactive=True)
                    # feedback_txt.change(
                    #     fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                    #     inputs=[feedback_txt],
                    #     outputs=[feedback_button]
                    # )

                    # feedback_button.click(
                    #     fn=self.create_prompt,
                    #     inputs=[feedback_txt],
                    #     outputs=[],
                    #     concurrency_limit=None,
                    #     concurrency_id="feedback"
                    # ).then(
                    #     fn=lambda: (gr.Textbox(value="", interactive=False), gr.Button(interactive=False)),
                    #     inputs=[],
                    #     outputs=[feedback_txt, feedback_button]
                    # )
    
    def create_prompt(self):
        """
        Handle prompt creation
        """
        #logger.debug(f"User Feedback from {session_state.session}: {text}")
        try:
            pass
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            pass
            
        gr.Info("""
        Your Prompt has been created. Switch now to the "Text" - Tab to start the image generation.
        """)