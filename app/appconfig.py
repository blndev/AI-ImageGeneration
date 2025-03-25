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
        self.modelconfig_json = os.getenv("MODELCONFIG", "./modelconfig.json")
        self.selected_model = os.getenv("GENERATION_MODEL", "default")
        self.model_cache_dir = os.getenv("MODEL_DIRECTORY", "./models/")

        self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
        self.token_enabled = self.initial_token > 0
        self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))

        self.allow_upload = self.getbool("ALLOW_UPLOAD", False)
        self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)

        if self.output_directory is None:
            self.user_feedback_filestorage = "./output/feedback.txt"
        else:
            self.user_feedback_filestorage = os.path.join(
                self.output_directory, get_date_subfolder(), "feedback.txt"
            )

        self.feature_prompt_magic_enabled = self.getbool("PROMPTMAGIC", False)

        self.NO_AI = self.getbool("NO_AI", False)
        self.GPU_ALLOW_ATTENTION_SLICING = self.getbool(
            "GPU_ALLOW_ATTENTION_SLICING", False
        )
        self.GPU_ALLOW_XFORMERS = self.getbool("GPU_ALLOW_XFORMERS", False)
        self.GPU_ALLOW_MEMORY_OFFLOAD = self.getbool("GPU_ALLOW_MEMORY_OFFLOAD", False)
