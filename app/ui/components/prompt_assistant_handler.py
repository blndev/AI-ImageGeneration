from datetime import datetime
import os
import gradio as gr
import logging
from app import SessionState
from app.appconfig import AppConfig
from app.utils.singleton import singleton
from app.analytics import Analytics
from .image_generator import ImageGenerationHandler

# Set up module logger
logger = logging.getLogger(__name__)


@singleton
class PromptAssistantHandler:
    """
    Handler for user feedback functionality
    """

    def __init__(self, analytics: Analytics, image_generator: ImageGenerationHandler):
        self.analytics = analytics
        self.image_generator = image_generator

        # helpers
        self._human_style_objects = ["Human", "Robot", "Alien", "Fairy", "Woman", "Girl", "Man", "Boy"]

    def _load_ui_dependencies(self):
        """load configuration values for the ui from external sources or generate them"""
        try:
            # TODO: Place things like object and location here
            self.feedback_message = ""
        except Exception as e:
            logger.error(f"Error while loading assistant dependencies: {e}")

    def toggle_people_details_visibility(self, is_checked):
        return gr.Group(visible=is_checked)

    def create_interface_elements(self, session_state):
        self._load_ui_dependencies()

        with gr.Row():
            gr.Markdown("This section helps you to create a prompt. Just select the elements you want to have in the image")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Choose Main Object")
                # chkFemale = gr.Checkbox(label="Female", value=True)
                gr_image_object = gr.Dropdown(choices=["Human", "Fairy", "Alien", "Robot", "Dog", "Bird",
                                                       "Cow", "Custom"], label="Main Object", interactive=True, allow_custom_value=True)
                gr_txt_custom_object = gr.Textbox(
                    value="",
                    label="Custom Object",
                    placeholder="Whatever you can imagine",
                    visible=False, interactive=True
                )

                gr_age = gr.Slider(interactive=True, minimum=5, maximum=85, value=25,
                                   step=10, label="Age", info="Choose the Age of that Object")

                with gr.Group(visible=True) as human_details_group:

                    gr_gender = gr.Radio(["Female", "Male"], value="Female")
                    gr_body_details = gr.Dropdown(
                        ["Random"],
                        value=["Random"],
                        multiselect=True, label="Body details", allow_custom_value=True,
                        interactive=True
                    )
                    gr_facial_expression = gr.Dropdown(
                        ["Smiling", "Neutral", "Angry", "Sad"],
                        value="Smiling",
                        multiselect=False, label="Facial expression", allow_custom_value=True,
                        interactive=True
                    )
                    gr_pose = gr.Dropdown(
                        ["Posing", "Sitting", "Walking", "Dynamic Pose"],
                        value="Posing",
                        multiselect=False, label="Pose", allow_custom_value=True,
                        interactive=True
                    )
                    gr_cloth = gr.Dropdown(
                        ["Shorts", "Tank Top", "Casual", "Swimwear", "Sunglasses"],
                        value=["Casual", "Sunglasses"],
                        multiselect=True, label="Clothes", allow_custom_value=True,
                        interactive=True
                    )

            with gr.Column():
                gr.Markdown("Environment")
                gr_location = gr.Dropdown(
                    ["Random", "Home", "Backyard", "Forest", "Beach", "Castle", "Photo Studio"],
                    value="Random",
                    multiselect=False, label="Location", allow_custom_value=True,
                    interactive=True
                )

        gr_image_object.change(
            fn=lambda o: gr.Textbox(visible=(o == "Custom")),
            inputs=gr_image_object,
            outputs=gr_txt_custom_object
        ).then(
            fn=lambda o: gr.Group(visible=(o in self._human_style_objects)),
            inputs=gr_image_object,
            outputs=human_details_group
        ).then(  # TODO move to dedicated function and use it also for other change handlers,
            # TODO: add a caching for the combinations of g o a in a dict to prevent alwways regenerations
            fn=lambda o, a, g: gr.Dropdown(choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "cloths", self.image_generator.prompt_refiner.create_better_words_for(f"{g} {o} age {a}"))),
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_cloth
        ).then(
            fn=lambda o, a, g: gr.Dropdown(choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "locations", self.image_generator.prompt_refiner.create_better_words_for(f"{g} {o} age {a}"))),
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_location
        ).then(
            fn=lambda o, a, g: gr.Dropdown(choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "individual body details (examples:  green eyes, dark hair, tall, fat)", self.image_generator.prompt_refiner.create_better_words_for(f"{g} {o} age {a}"))),
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_body_details
        )

        with gr.Row():
            gr_button_create_image = gr.Button("Create Image", visible=True, interactive=True)

        with gr.Row():
            generated_image = gr.Image()  # TODO: just temporary until we get the gallery referenced

            # TODO reference also aspect_ratio etc
            # IDEA: create_image should write the prompt to "assistant_prompt" (a hidden text field) and then call FIXME
            # the ui#_create_image from gradio_ui, that will solve aspect, gallery etc.
            # TODO prompt magic prompt must be copied to standrad prompt field
            gr_button_create_image.click(
                fn=self.create_image,
                inputs=[
                    session_state,
                    gr_image_object,
                    gr_txt_custom_object,
                    gr_gender,
                    gr_age,
                    gr_location,
                    gr_facial_expression,
                    gr_body_details,
                    gr_pose,
                    gr_cloth],
                outputs=[generated_image],
                concurrency_limit=None,
                concurrency_id=""
            )  # .then(
            #     fn=lambda: (gr.Button(interactive=False)),
            #     inputs=[],
            #     outputs=[gr_button_create_image]
            # )

    def create_image(
            self,
            gr_state,
            image_object,
            txt_custom_object,
            gender,
            age,
            location,
            gr_facial_expression,
            gr_body_details,
            gr_pose,
            gr_cloth,
            progress=gr.Progress()):
        """
        Handle prompt creation
        """
        # logger.debug(f"User Feedback from {session_state.session}: {text}")
        try:
            session_state = SessionState.from_gradio_state(gr_state)

            if image_object != "Custom": txt_custom_object = image_object
            txtage = "average"
            if age > 60: txtage = "old"
            if age > 80: txtage = "very old"
            if age < 30: txtage = "younger"
            if age < 20: txtage = "young"
            if age < 10: txtage = "very young"
            humanprompt = ""
            if txt_custom_object in self._human_style_objects:
                humanprompt = f"{gr_facial_expression} with {gr_body_details}, wearing {gr_cloth}, {gr_pose}"

            # optimize wording
            better_object = f"{txtage} {gender} {txt_custom_object}"
            if self.image_generator.prompt_refiner:
                better_object = self.image_generator.prompt_refiner.create_better_words_for(better_object)

            prompt = f"photo of a {better_object}, {humanprompt}, in {location} location"
            logger.debug(f"Assistant Prompt: {prompt}")
            image, _, _ = self.image_generator.generate_images(
                progress=progress,
                session_state=session_state,
                prompt=prompt,
                neg_prompt="",
                aspect_ratio="Square",
                user_activated_promptmagic=True,  # always use prompt magic to make a nice prompt from such keywords
                image_count=1  # TODO: get from external
            )
            return image[0]
        except Exception as e:
            logger.error(f"Error creating assistant based image: {e}")
            gr.Error(e)
