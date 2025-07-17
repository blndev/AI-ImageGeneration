from datetime import datetime
import json
import os
import random
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

    def __init__(self, analytics: Analytics, config: AppConfig, image_generator: ImageGenerationHandler):
        self.analytics = analytics
        self.config = config
        self.image_generator = image_generator

        # helpers
        self._human_style_objects = ["Clown", "Woman", "Man", "Robot", "Alien", "Fairy", "Woman", "Girl", "Boy"]
        self._suggestions_cache = {}
        self._initialize_database_assistant_suggestions()

    def _initialize_database_assistant_suggestions(self):
        self._created_images_data = {}

        try:
            # check maybe it's better to add the data folder as well
            self.__suggestions_cache_path = os.path.join(self.config.output_directory, "assistant_suggestions.json")
            if os.path.exists(self.__suggestions_cache_path):
                with open(self.__suggestions_cache_path, "r") as f:
                    self._suggestions_cache.update(json.load(f))
            logger.info(f"Initialized assistant_suggestions from '{self.__suggestions_cache_path}'")
        except Exception as e:
            logger.error(f"Error while loading assistant_suggestions.json: {e}")

    def _load_ui_dependencies(self):
        """load configuration values for the ui from external sources or generate them"""
        try:
            pass
        except Exception as e:
            logger.error(f"Error while loading assistant dependencies: {e}")

    def toggle_people_details_visibility(self, is_checked):
        return gr.Group(visible=is_checked)

    def _is_image_human_style(self, object_description):
        # check if the object description is pointing somehow to human style objects
        # this is used to optimize the generated prompt and show ui details
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

    def _create_better_words_for(self, subject, age):
        better = ""
        text_age = ""

        # required as in the logs where issues from gradio which references NoneType for age (could be timing issue in JS)
        if not age: age = 25
        if not subject: subject = "clown"

        if age > 50: text_age = "old"
        if age > 70: text_age = "very old"
        if age < 20: text_age = "young"
        if age < 10: text_age = "very young"

        if self.image_generator.prompt_refiner:
            better = self.image_generator.prompt_refiner.create_better_words_for(f"{subject} age {age}")
            better = better.replace("[", "")
            if not self._is_image_human_style(better):
                better = f"{text_age} {subject}"
        else:
            # fallback
            if "woman" in subject.lower():
                subject = "Girl" if age < 18 else "Woman"
            elif "man" in subject.lower():
                subject = "Boy" if age < 18 else "Man"

            better = f"{text_age} {subject}"

        return better

    def create_suggestions_for_assistant(self, main_object_of_image) -> tuple:
        logger.debug(f"get suggestions for '{main_object_of_image}'")
        cloths = []
        locations = []
        body_details = []
        stereotypes = []
        retVal = self.__create_suggestions_ui_retval(cloths, locations, body_details, stereotypes)

        if main_object_of_image in self._suggestions_cache:
            logger.debug("cache used for suggestions")
            o = self._suggestions_cache[main_object_of_image]
            cloths = o["cloths"]
            locations = o["locations"]
            body_details = o["body_details"]
            stereotypes = o["stereotypes"]
            return self.__create_suggestions_ui_retval(cloths, locations, body_details, stereotypes)

        if self.image_generator.prompt_refiner:
            cloths = self.image_generator.prompt_refiner.create_list_of_x_for_y("cloths weared", main_object_of_image)
            locations = self.image_generator.prompt_refiner.create_list_of_x_for_y("locations", main_object_of_image, element_count=20)
            body_details = self.image_generator.prompt_refiner.create_list_of_x_for_y(
                "well defined body details (examples:  green eyes, dark hair, tall, makeup, eyeliners)",
                main_object_of_image)
            stereotypes = self.image_generator.prompt_refiner.create_list_of_x_for_y("stereotypes, jobs or roles", main_object_of_image)
            retVal = self.__create_suggestions_ui_retval(cloths, locations, body_details, stereotypes)
            try:
                # caching for the combinations of g o a in a dict to prevent always regenerations
                # it can later also be used to offer suggestions without an llm
                o = {}
                o["cloths"] = cloths
                o["locations"] = locations
                o["body_details"] = body_details
                o["stereotypes"] = stereotypes
                self._suggestions_cache[main_object_of_image] = o
                with open(self.__suggestions_cache_path, "w") as f:
                    json.dump(self._suggestions_cache, f, indent=4)
            except Exception as e:
                logger.error(f"Error while saving {self.__suggestions_cache_path}: {e}")

        return retVal

    def __create_suggestions_ui_retval(self, cloths, locations, body_details, stereotypes):
        retVal = (gr.Dropdown(value=[], choices=cloths),
                  gr.Dropdown(value="", choices=locations),
                  gr.Dropdown(value=[], choices=body_details),
                  gr.Dropdown(value="", choices=stereotypes))

        return retVal

    def _list_to_simple_string(self, source_list: list) -> str:
        string = ""
        for element in source_list:
            string += f"{element},"
        return string

    def create_interface_elements(self, session_state, assistant_prompt):
        self._load_ui_dependencies()

        with gr.Row():
            gr.Markdown("""This Assistant helps you to create an image.
                        Just select the elements you want to have in the image, or write missing things into the fields.
                        To get a more precice output, copy the generated prompt from Prompt Magic in the "Advanced"
                        section into the free-style prompt box.
                        Be aware that we make your prompts silent safe for work or censor unsafe content until *you* turn this protection off.
                        """)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(visible=True):
                    gr.Markdown("General Settings")
                    # chkFemale = gr.Checkbox(label="Female", value=True)
                    gr_image_object = gr.Dropdown(
                        choices=["Clown", "Woman", "Man", "Fairy", "Alien", "Robot", "Dog", "Bird",
                                 "Cow", "Unicorn"],
                        label="Main Object",
                        info="add anything you can Imagine",
                        interactive=True,
                        allow_custom_value=True)

                    gr_age = gr.Slider(
                        interactive=True,
                        minimum=1,
                        maximum=100,
                        value=25,
                        step=2,
                        label="Age",
                        info="Choose the Age of that Object")

                    # used as main input for the prompt and ai suggestions
                    gr_txt_custom_object = gr.Textbox(
                        value=f" {gr_age.value} year old {gr_image_object.value}",  # based on defaults of the other controls
                        label="Optimized Object description (read only)",
                        placeholder="",
                        visible=False, interactive=False
                    )

                    gr_style = gr.Dropdown(
                        ["Random",
                            "Photo",
                            "Full-Body Portrait",
                            "Minimalistic",
                            "Monochrome",
                            "Tribal",
                            "Futurism", "Cyberpunk", "Cybernetic Human", "Cybernetic Robot",
                            "PopArt", "Comic",
                            "Gothic", "Neon",
                            "Painting",
                            "Line Art",
                            "Abstract drawing"],
                        value="Photo",
                        multiselect=False,
                        label="Style",
                        info="You can add a whole style and using'{prompt}' as placeholder",
                        allow_custom_value=True,
                        interactive=True
                    )

            with gr.Column(scale=2):
                with gr.Group(visible=True):
                    gr.Markdown("Environment and Details")
                    with gr.Row():
                        with gr.Column():
                            gr_location = gr.Dropdown(
                                ["", "Random", "Home", "Backyard", "Forest", "Beach", "Castle", "Photo Studio", "Pizza Restaurant", "Moon"],
                                value="",
                                multiselect=False,
                                label="Location",
                                allow_custom_value=True,
                                interactive=True
                            )
                        with gr.Column():
                            gr_stereotype = gr.Dropdown(
                                ["", "Developer", "Analyst", "CEO", "Doctor", "Nurse", "Wizzard", "Witch", "Teacher", "Student", "Schoolgirl"],
                                value="",
                                multiselect=False, label="Stereotype", allow_custom_value=True,
                                interactive=True
                            )

                    with gr.Row():
                        gr_body_details = gr.Dropdown(
                            ["red Hair", "curly Hair", "perfect Face", "blue Eyes", "red Lipstick", "intense Makeup"],
                            value=[],
                            multiselect=True, label="Body details", allow_custom_value=True,
                            interactive=True
                        )
                    with gr.Row():
                        gr_cloth = gr.Dropdown(
                            ["Shorts", "Tank Top", "Casual", "Swimwear", "Sunglasses", "Collar", "no clothes"],
                            value=["Casual", "Sunglasses"],
                            multiselect=True, label="Wearing", allow_custom_value=True,
                            interactive=True
                        )
                    with gr.Row():
                        gr_button_update_suggestions = gr.Button(
                            value="Update suggestions for Environment and Details",
                            visible=self.image_generator.prompt_refiner is not None,
                            size="md")

            with gr.Column(scale=1):
                with gr.Group(visible=False) as human_details_group:
                    gr.Markdown("Advanced Details")
                    gr_facial_expression = gr.Dropdown(
                        ["", "Smiling", "Neutral", "Angry", "Sad", "Shy", "Surprised"],
                        value="Smiling",
                        multiselect=False, label="Facial expression", allow_custom_value=True,
                        interactive=True
                    )
                    gr_pose = gr.Dropdown(
                        ["", "Posing", "Sitting", "Walking", "Dancing", "Eating", "Watching"],
                        value="Posing",
                        multiselect=False, label="Pose or Action", allow_custom_value=True,
                        interactive=True
                    )
                with gr.Group():
                    gr_token_info = gr.Text(value="", container=True, show_label=False)
                    gr_button_create_image = gr.Button("Create Image", visible=True, interactive=True, variant="primary")

            ################################################
            # handle recreation of target object description
            ################################################
            gr_age.change(
                fn=self._create_better_words_for,
                inputs=[gr_image_object, gr_age],
                outputs=gr_txt_custom_object
            )
            gr_image_object.change(
                fn=self._create_better_words_for,
                inputs=[gr_image_object, gr_age],
                outputs=gr_txt_custom_object
            )

            ################################################
            # switch visibility of human specific properties
            gr_txt_custom_object.change(
                fn=lambda o: gr.Group(visible=(self._is_image_human_style(o))),
                inputs=gr_txt_custom_object,
                outputs=human_details_group
            )

            ######################################################################
            # reload or regenerate all suggestions for location, body details etc.
            gr_button_update_suggestions.click(
                fn=self.create_suggestions_for_assistant,
                inputs=[gr_txt_custom_object],
                outputs=[gr_cloth, gr_location, gr_body_details, gr_stereotype],
                concurrency_id="llm",
                concurrency_limit=10
            )

        ####################################################
        # start of final event handlers
        ####################################################
        gr_button_create_image.click(
            fn=lambda: (gr.Button(interactive=False)),
            inputs=[],
            outputs=gr_button_create_image
        ).then(
            fn=self.create_image,
            inputs=[
                gr_style,
                gr_txt_custom_object,
                gr_location,
                gr_facial_expression,
                gr_body_details,
                gr_pose,
                gr_stereotype,
                gr_cloth],
            outputs=[assistant_prompt],
            concurrency_limit=None,
            concurrency_id=""
        )
        # we need to return the button to deactivate him from external and the label which
        # receives the token info and the generation status
        return gr_button_create_image, gr_token_info

    def create_image(
            self,
            style: str,
            object_description: str,
            location: str,
            gr_facial_expression: str,
            gr_body_details: list,
            gr_pose: str,
            stereotype: str,
            gr_cloth: list
    ):
        """
        Handle prompt creation and trigger image creation via assistant_prompt_change
        """
        # logger.debug(f"User Feedback from {session_state.session}: {text}")
        try:
            humanprompt = ""
            # dedicated to humans
            if self._is_image_human_style(object_description):
                face = ""
                if gr_facial_expression and len(gr_facial_expression) > 0:
                    face = f"is {gr_facial_expression},"

                pose = ""
                if gr_pose:
                    pose = f"and {gr_pose}"
                humanprompt = f"{face} {pose}."

            # for all objects
            if location:
                location = f"at {location} location,"

            if stereotype:
                stereotype = f"(stereotype: {stereotype})"

            # handling lists
            body = ""
            if gr_body_details is not None and len(gr_body_details) > 0:
                body = f"with {self._list_to_simple_string(gr_body_details)}"

            cloth = f"wearing {self._list_to_simple_string(gr_cloth)}," if gr_cloth and len(gr_cloth) > 0 else ""

            # compose prompt
            prompt = f"{location} a {object_description} {body} {cloth} {humanprompt} {stereotype}"

            # before apply the style, the prompt must run trough prompt magic
            # this marker is later used to extract the prompt
            prompt = f"{self.config.promptmarker}{prompt}{self.config.promptmarker}"

            if style == "Random": style = random.choice(
                ["PopArt", "Photo", "Futurism", "Gothic", "Disco", "Minimalistic",
                 "Monochrome", "Tribal", "Neon", "Cyberpunk", "Comic", "Line Art"]
            )

            # TODO Vx - IDEA place styles in dedicated function, load styles from files or modelconfig
            if "PopArt" in style: prompt = f"pop art collage style with red lipstick, comic style speech bubbles style: {prompt}"
            elif "Photo" in style: prompt = f"Professional photo {prompt} . large depth of field, deep depth of field, highly detailed"
            elif "Futurism" in style: prompt = f"futuristic cityscape, futurism: {prompt} . flying cars, dynamic lines and vibrant colors,"
            elif "Gothic" in style: prompt = f"gothic style {prompt} . dark, mysterious, haunting, dramatic, ornate, detailed"
            elif "Abstract drawing" in style: prompt = f"abstract expressionist painting: {prompt} . energetic brushwork, bold colors, abstract forms, expressive, emotional"
            elif "Disco" in style: prompt = f"disco-themed {prompt} . vibrant, groovy, retro 70s style, shiny disco balls, neon lights, dance floor, highly detailed"
            elif "Minimalistic" in style: prompt = f"minimalist style {prompt} . simple, clean, uncluttered, modern, elegant"
            elif "Monochrome" in style: prompt = f"monochrome {prompt} . black and white, contrast, tone, texture, detailed"
            elif "Tribal" in style: prompt = f"tribal style {prompt} . indigenous, ethnic, traditional patterns, bold, natural colors, highly detailed"
            elif "Neon" in style: prompt = f"neon noir {prompt} . cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed"
            elif "Cyberpunk" in style: prompt = f"biomechanical cyberpunk {prompt} . cybernetics, human-machine fusion, dystopian, organic meets artificial, dark, intricate, highly detailed"
            elif "Cybernetic Human" in style: prompt = f"cybernetic style {prompt} . futuristic, technological, cybernetic enhancements, robotics, artificial intelligence themes"
            elif "Cybernetic Robot" in style: prompt = f"cybernetic robot {prompt} . android, AI, machine, metal, wires, tech, futuristic, highly detailed"
            elif "Painting" in style: prompt = f"impressionist painting {prompt} . loose brushwork, vibrant color, light and shadow play, captures feeling over form"
            elif "Comic" in style: prompt = f"comic of {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
            elif "Line Art" in style: prompt = f"line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
            elif "{prompt}" in style: prompt = style.replace("{prompt}", prompt)
            else: prompt = f"{style} style: {prompt}"

            logger.debug(f"Assistant Prompt: {prompt}")

            # trick to create a change event on the textbox, as the prompt is always striped, we can add a random amount of spaces
            spaces = ""
            for _ in range(random.randint(1, 55)):
                spaces += " "
            # logger.warning(len(spaces))
            return prompt + spaces
        except Exception as e:
            logger.error(f"Error creating assistant based image: {e}")
            gr.Error(e)
