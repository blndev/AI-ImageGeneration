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

    def _is_image_human_style(self, object_description):
        # TODO replace with intelligence and use list as fallback
        object_description = object_description.lower()
        value = object_description in self._human_style_objects \
            or "man" in object_description \
            or "woman" in object_description \
            or "boy" in object_description \
            or "girl" in object_description \
            or "teen" in object_description \
            or "child" in object_description
        logger.debug(f"Is the object '{object_description}' a Human? {value}")
        return value

    def _create_better_words_for(self, subject, age, gender):
        better = ""
        text_age = ""
        if age > 50: text_age = "old"
        if age > 70: text_age = "very old"
        if age < 20: text_age = "young"
        if age < 10: text_age = "very young"

        if self.image_generator.prompt_refiner:
            better = self.image_generator.prompt_refiner.create_better_words_for(f"{gender} {subject} age {age}")
            if not self._is_image_human_style(better):
                better = f"{text_age} {subject}"
        else:
            # fallback
            if subject == "Human":
                if gender.lower() == "female":
                    subject = "Girl" if age < 18 else "Woman"
                else:
                    subject = "Boy" if age < 18 else "Man"

            elif self._is_image_human_style(subject):
                subject = f"{gender} {subject}"
            else:
                subject = f"{subject}"
            better = f"{text_age} {subject}"

        return better

    def _list_to_simple_string(self, source_list: list) -> str:
        string = ""
        for element in source_list:
            string += f"{element},"
        return string

    def create_interface_elements(self, session_state):
        self._load_ui_dependencies()

        with gr.Row():
            gr.Markdown("""Our Assistanthelps you to create an image. 
                        Just select the elements you want to have in the image, or write missing things into the fields.
                        To get more precice output, deactivate Prompt Magic in the "Advanced" section.
                        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("Choose Main Object")
                # chkFemale = gr.Checkbox(label="Female", value=True)
                gr_image_object = gr.Dropdown(
                    choices=["Human", "Fairy", "Alien", "Robot", "Dog", "Bird",
                             "Cow", "Custom"],
                    label="Main Object",
                    info="add anything you Imagine",
                    interactive=True,
                    allow_custom_value=True)
                # used as main input for the prompt and ai suggestions
                gr_txt_custom_object = gr.Textbox(
                    value="",
                    label="Optimized Object description (read only)",
                    placeholder="",
                    visible=False, interactive=False
                )

                gr_age = gr.Slider(interactive=True, minimum=1, maximum=100, value=25,
                                   step=2, label="Age", info="Choose the Age of that Object")

                gr_button_update_suggestions = gr.Button(
                    value="Update suggestions for Location, Cloth and more",
                    visible=self.image_generator.prompt_refiner is not None
                )

            with gr.Column():
                gr.Markdown("Environment and Styles")
                gr_location = gr.Dropdown(
                    ["Random", "Home", "Backyard", "Forest", "Beach", "Castle", "Photo Studio", "Pizza Restaurant", "Moon"],
                    value="Random",
                    multiselect=False, label="Location", allow_custom_value=True,
                    interactive=True
                )
                gr_style = gr.Dropdown(
                    ["Random",
                     "Photo",
                     "Full-Body Portrait",
                     "Minimalistic",
                     "Monochrome",
                     "Tribal",
                     "Futurism", "Cyperpunk", "Cybernetic Human", "Cybernetic Robot",
                     "PopArt", "Comic",
                     "Gothic", "Neon",
                     "Painting",
                     "abstract expressionist"],
                    value="Random",
                    multiselect=False,
                    label="Style",
                    info="You can add a whole style and using'{prompt}' as placeholder",
                    allow_custom_value=True,
                    interactive=True
                )

        with gr.Row():

            with gr.Group(visible=True) as human_details_group:

                gr_gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                gr_body_details = gr.Dropdown(
                    ["Random", "perfect Face", "blue Eyes", "red Lipstick", "intense Makeup"],
                    value=["Random"],
                    multiselect=True, label="Body details", allow_custom_value=True,
                    interactive=True
                )
                gr_facial_expression = gr.Dropdown(
                    ["Smiling", "Neutral", "Angry", "Sad", "Shy", "Surprised"],
                    value="Smiling",
                    multiselect=False, label="Facial expression", allow_custom_value=True,
                    interactive=True
                )
                gr_pose = gr.Dropdown(
                    ["Posing", "Sitting", "Walking", "Dancing", "Eating", "Watching"],
                    value="Posing",
                    multiselect=False, label="Pose or Action", allow_custom_value=True,
                    interactive=True
                )
                gr_cloth = gr.Dropdown(
                    ["Shorts", "Tank Top", "Casual", "Swimwear", "Sunglasses", "no clothes"],
                    value=["Casual", "Sunglasses"],
                    multiselect=True, label="Wearing", allow_custom_value=True,
                    interactive=True
                )
                gr_stereotype = gr.Dropdown(
                    ["Random", "Developer", "Analyst", "CEO", "Doctor", "Nurse", "Wizzard", "Witch", "Teacher", "Student", "Schoolgirl"],
                    value="Random",
                    multiselect=False, label="Stereotype", allow_custom_value=True,
                    interactive=True
                )

        gr_age.change(
            fn=self._create_better_words_for,
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_txt_custom_object
        )
        gr_image_object.change(
            fn=self._create_better_words_for,
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_txt_custom_object
        )
        gr_gender.change(
            fn=self._create_better_words_for,
            inputs=[gr_image_object, gr_age, gr_gender],
            outputs=gr_txt_custom_object
        )

        gr_txt_custom_object.change(
            fn=lambda o: gr.Group(visible=(self._is_image_human_style(o))),
            inputs=gr_txt_custom_object,
            outputs=human_details_group
        )
        gr_button_update_suggestions.click(  # TODO move to dedicated function and use it also for other change handlers,
            # TODO: add a caching for the combinations of g o a in a dict to prevent alwways regenerations
            # TODO: keep selected values in the list
            fn=lambda o: gr.Dropdown(value=[], choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "cloths weared", o)),
            inputs=[gr_txt_custom_object],
            outputs=gr_cloth
        ).then(
            fn=lambda o, current: gr.Dropdown(value=None, choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "locations", o, element_count=20)),  # TODO .append(current)
            inputs=[gr_txt_custom_object, gr_location],
            outputs=gr_location
        ).then(
            fn=lambda o: gr.Dropdown(value=[], choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "individual body details (examples:  green eyes, dark hair, tall, lipstick, eyeliners)", o)),
            inputs=[gr_txt_custom_object],
            outputs=gr_body_details
        ).then(
            fn=lambda o, current: gr.Dropdown(value=None, choices=self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "stereotypes or jobs", o)),  # TODO .append(current)
            inputs=[gr_txt_custom_object, gr_stereotype],
            outputs=gr_stereotype
        )

        with gr.Row():
            gr_button_create_image = gr.Button("Create Image (99 creating left, 3 uncensored)",
                                               visible=True, interactive=True, variant="primary")
            # TODO add token count same as in freestyle box
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
                    gr_style,
                    gr_txt_custom_object,
                    gr_location,
                    gr_facial_expression,
                    gr_body_details,
                    gr_pose,
                    gr_stereotype,
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
            gr_state: SessionState,
            style: str,
            object_description: str,
            location: str,
            gr_facial_expression: str,
            gr_body_details: list,
            gr_pose: str,
            stereotype: str,
            gr_cloth: list,
            progress=gr.Progress()):
        """
        Handle prompt creation
        """
        # logger.debug(f"User Feedback from {session_state.session}: {text}")
        try:
            session_state = SessionState.from_gradio_state(gr_state)

            humanprompt = ""
            if self._is_image_human_style(object_description):
                body = ""
                if gr_body_details is not None and len(gr_body_details) > 0:
                    body = f"with {self._list_to_simple_string(gr_body_details)}"
                if stereotype:
                    stereotype = f"stereotype: {stereotype}"
                face = ""
                if gr_facial_expression and len(gr_facial_expression) > 0:
                    face = f"and prefect face with {gr_facial_expression} expression,"
                if gr_body_details and len(gr_body_details) > 0:
                    body = f"with {self._list_to_simple_string(gr_body_details)}"

                cloth = f"wearing {self._list_to_simple_string(gr_cloth)}" if len(gr_cloth) > 0 else ""

                pose = ""
                if gr_pose:
                    pose = f"while {gr_pose}"
                humanprompt = f"{body} {face} {cloth} {stereotype}"
            if location:
                location = "in {location} location,"
            prompt = f"a {object_description} {location} {humanprompt}"
            # TODO refactor in dedicated function, load styles from files or modelconfig
            if "PopArt" in style: prompt = f"pop art collage style with red lipstick, comic style speech bubbles style: {prompt}"
            elif "Photo" in style: prompt = f"iphone photo {prompt} . large depth of field, deep depth of field, highly detailed"
            elif "Futurism" in style: prompt = f"futuristic cityscape, futurism: {prompt} . flying cars, dynamic lines and vibrant colors,"
            elif "Gothic" in style: prompt = f"gothic style {prompt} . dark, mysterious, haunting, dramatic, ornate, detailed"
            elif "abstract expressionist" in style: prompt = f"abstract expressionist painting: {prompt} . energetic brushwork, bold colors, abstract forms, expressive, emotional"
            elif "Disco" in style: prompt = f"disco-themed {prompt} . vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed"
            elif "Minimalistic" in style: prompt = f"minimalist style {prompt} . simple, clean, uncluttered, modern, elegant"
            elif "Monochrome" in style: prompt = f"monochrome {prompt} . black and white, contrast, tone, texture, detailed"
            elif "Tribal" in style: prompt = f"tribal style {prompt} . indigenous, ethnic, traditional patterns, bold, natural colors, highly detailed"
            elif "Neon" in style: prompt = f"neon noir {prompt} . cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed"
            elif "Cyberpunk" in style: prompt = f"biomechanical cyberpunk {prompt} . cybernetics, human-machine fusion, dystopian, organic meets artificial, dark, intricate, highly detailed"
            elif "Cybernetic Human" in style: prompt = f"cybernetic style {prompt} . futuristic, technological, cybernetic enhancements, robotics, artificial intelligence themes"
            elif "Cybernetic Robot" in style: prompt = f"cybernetic robot {prompt} . android, AI, machine, metal, wires, tech, futuristic, highly detailed"
            elif "Painting" in style: prompt = f"impressionist painting {prompt} . loose brushwork, vibrant color, light and shadow play, captures feeling over form"
            elif "Comic" in style: prompt = f"comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
            elif "{prompt}" in style: prompt = style.replace("{prompt}", prompt)
            else: prompt = f"{style} style: {prompt}"

            logger.debug(f"Assistant Prompt: {prompt}")
            image, _, _ = self.image_generator.generate_images(
                progress=progress,
                session_state=session_state,
                prompt=prompt,
                neg_prompt="",
                aspect_ratio="Square",
                user_activated_promptmagic=True,  # TODO get from prompt magic setting in advanced ui
                image_count=1  # TODO: get from external
            )
            return image[0]
        except Exception as e:
            logger.error(f"Error creating assistant based image: {e}")
            gr.Error(e)
