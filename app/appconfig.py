import os
from distutils.util import strtobool
from app.utils.fileIO import get_date_subfolder
from .utils.singleton import singleton

import logging
logger = logging.getLogger(__name__)


@singleton
class AppConfig:
    def __init__(self):
        self.refresh()

    def getbool(self, key, default=False):
        return bool(strtobool(os.getenv(key, str(default))))

    def refresh(self):
        logger.info("refresh config")
        self.app_environment = os.getenv("IMGGEN_ENVIRONMENT", "unkown")

        self.GRADIO_SHARED = self.getbool("GRADIO_SHARED", False)
        self.modelconfig_json = os.getenv("MODELCONFIG", "./modelconfig.json")

        self.selected_model = os.getenv("GENERATION_MODEL", "default")
        self.model_cache_dir = os.getenv("MODEL_DIRECTORY", "./models/")
        self.free_memory_after_minutes_inactivity = int(os.getenv("FREE_MEMORY_AFTER_MINUTES_INACTIVITY", 15))

        self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
        self.feature_generation_credits_enabled = self.initial_token > 0

        self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))

        self.feature_sharing_links_new_token_per_image = int(os.getenv("FEATURE_SHARING_LINK_NEW_TOKEN", 0))
        self.feature_sharing_links_enabled = self.feature_sharing_links_new_token_per_image > 0

        self.feature_upload_images_token_reward = int(os.getenv("FEATURE_UPLOAD_IMAGE_NEW_TOKEN", 0))
        self.feature_upload_images_for_new_token_enabled = self.feature_upload_images_token_reward > 0
        self.feature_allow_nsfw = self.getbool("FEATURE_ALLOW_NSFW", False)

        self.save_generated_output = (os.getenv("OUTPUT_DIRECTORY", None) is not None)
        self.output_directory = os.getenv("OUTPUT_DIRECTORY", "./output/")

        self.user_feedback_filestorage = os.path.join(
            self.output_directory, "feedback.txt"
        )

        self.feature_prompt_magic_enabled = self.getbool("PROMPTMAGIC", False)
        self.promptmarker = "#!!#"  # used to identify the real prompt in a style to avoud prompt magic overwrite of styles
        self.NO_AI = self.getbool("NO_AI", False)
