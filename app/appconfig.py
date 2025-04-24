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

    # TODO: load from file instead of env for most keys
    # TODO: make it more reliable
    def refresh(self):
        logger.info("refresh config")
        self.GRADIO_SHARED = self.getbool("GRADIO_SHARED", False)
        self.modelconfig_json = os.getenv("MODELCONFIG", "./modelconfig.json")
        self.selected_model = os.getenv("GENERATION_MODEL", "default")
        self.model_cache_dir = os.getenv("MODEL_DIRECTORY", "./models/")
        self.free_memory_after_minutes_inactivity = int(os.getenv("FREE_MEMORY_AFTER_MINUTES_INACTIVITY", 15))

        self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
        self.token_enabled = self.initial_token > 0
        self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))

        self.feature_sharing_link_new_token = int(os.getenv("FEATURE_SHARING_LINK_NEW_TOKEN", 0))
        self.feature_sharing_links_enabled = self.feature_sharing_link_new_token > 0

        self.feature_upload_images_token_reward = int(os.getenv("FEATURE_UPLOAD_IMAGE_NEW_TOKEN", 0))
        self.feature_upload_images_for_new_token_enabled = self.feature_upload_images_token_reward > 0

        self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)

        if self.output_directory is None:
            self.user_feedback_filestorage = "./output/feedback.txt"
            self.feature_upload_images_for_new_token_enabled = False
        else:
            self.user_feedback_filestorage = os.path.join(
                self.output_directory, get_date_subfolder(), "feedback.txt"
            )

        self.feature_prompt_magic_enabled = self.getbool("PROMPTMAGIC", False)

        self.NO_AI = self.getbool("NO_AI", False)
        self.GPU_ALLOW_ATTENTION_SLICING = self.getbool("GPU_ALLOW_ATTENTION_SLICING", False)
        self.GPU_ALLOW_XFORMERS = self.getbool("GPU_ALLOW_XFORMERS", False)
        self.GPU_ALLOW_MEMORY_OFFLOAD = self.getbool("GPU_ALLOW_MEMORY_OFFLOAD", False)
