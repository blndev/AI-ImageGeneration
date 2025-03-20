
import os, logging
from distutils.util import strtobool
from app.utils.fileIO import get_date_subfolder
from .utils.singleton import singleton

@singleton
class AppConfig:
    def __init__(self):
        self.refresh()

    def getbool(self, key, default=False):
        return bool(strtobool(os.getenv(key,str(default))))
    
    #TODO: load from file instead of env for most keys
    def refresh(self):
        self.selected_model = os.getenv("GENERATION_MODEL", )
        
        self.initial_token = int(os.getenv("INITIAL_GENERATION_TOKEN", 0))
        self.token_enabled = self.initial_token > 0
        self.new_token_wait_time = int(os.getenv("NEW_TOKEN_WAIT_TIME", 10))

        self.allow_upload = self.getbool("ALLOW_UPLOAD", False)
        self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)

        if self.output_directory == None:
            self.user_feedback_filestorage = "./output/feedback.txt"
        else:
            self.user_feedback_filestorage = os.path.join(self.output_directory, get_date_subfolder(),"feedback.txt")
        