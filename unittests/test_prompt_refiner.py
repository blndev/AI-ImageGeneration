import unittest
import random, uuid, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.validators import PromptRefiner

class TestGradioUISessionState(unittest.TestCase):
    def setUp(self):
        self.prompt_refiner = PromptRefiner()

        self.nsfw_prompts = [
             "naked image of a man and a woman",
             "image of a woman without clothes",
             "show her ass in the mirror",
             #"killed people in a house",
             #"na**d people", difficult to detect, same as p0rn FIXME
             "bloody brain",
        ]

        self.sfw_prompts = [
             "a bear and a dog",
             "fighting chicken",
             "happy family",
             "sad crying girl",
             "tierd man ",
             "woman in underwear",
             "couple on the beach, she wears a bikini",
             "a lingerie model in a photo studio"
        ]

    def test_replace_nsfw(self):
        """Test Replace NSFW from prompt."""

        for nsfw_prompt in self.nsfw_prompts:
            sfw_prompt = self.prompt_refiner.replace_nsfw(nsfw_prompt)
            is_nsfw, msg = self.prompt_refiner.contains_nsfw(sfw_prompt)
            print(f"Original prompt: {nsfw_prompt}\n SFW Prompt: {sfw_prompt}\n{msg}\n--------------------")
            self.assertFalse(is_nsfw, f"prompt '{sfw_prompt}' dosent contain nsfw but answer is '{msg}'")

        #self.assertIsNotNone(answer.get("human"), "there should be a human in this image")

    def test_nsfw_detection(self):
        """Test Prompt Contains NSFW."""

        for prompt in self.nsfw_prompts:
            is_nsfw, msg = self.prompt_refiner.contains_nsfw(prompt)
            print(f"---------------------\nPrompt: '{prompt}'\nResponse: {msg}\n--------------------")
            self.assertTrue(is_nsfw, f"prompt '{prompt}' should contain nsfw but answer is '{msg}'")

    def test_sfw_detection(self):
        """Test SFW Prompt accepted."""

        for prompt in self.sfw_prompts:
            is_nsfw, msg = self.prompt_refiner.contains_nsfw(prompt)
            print(f"{prompt}\n{msg}\n--------------------")
            self.assertFalse(is_nsfw, f"prompt '{prompt}' dosent contain nsfw but answer is '{msg}'")
