
import os
import threading
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from ollama import Client

logger = logging.getLogger(__name__)
# no need for Singleton, as the Ollama Server takes care for model loading


class PromptRefiner():
    """
    Analyze the prompts and optimize it
    the optimization includes also applying rules
    """

    def __init__(self):
        logger.info("Initializing PromptRefiner")
        self.thread_lock = threading.Lock()
        # self.model ="llama3.2" #prefered, but partial issues with prompt enhance
        self.model = os.getenv("OLLAMA_MODEL", "llava").strip()
        self.ollama_server = os.getenv("OLLAMA_SERVER", None)

        self.llm = None
        try:
            olc = Client(self.ollama_server)
            olc.pull(self.model)
            self.llm = ChatOllama(
                model=self.model,
                server=self.ollama_server,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"Initialize llm for PromptRefiner failed {e}")

        try:
            self.llm_creative = ChatOllama(
                model=self.model,
                server=self.ollama_server,
                temperature=0.6,
            )
        except Exception as e:
            logger.error(f"Initialize llm for PromptRefiner failed {e}")

    def validate_refiner_is_ready(self) -> bool:
        validation = True
        try:
            if self.llm is None: raise Exception("llm not initialized")
            messages = [
                SystemMessage("You answer only with yes and no."),
                HumanMessage("Are you ready to work for me?"),
            ]
            ai_msg = self.llm.invoke(messages)
            logger.debug(f"Validate PromptRefiner-ready state: {ai_msg.content}")
            if "yes" not in ai_msg.content.lower():
                raise Exception(f"LLM is not able to work: {ai_msg.content}")
        except Exception as e:
            logger.warning(f"Validation of PromptRefiner failed with {e}")
            validation = False

        return validation

    def is_safe_for_work(self, prompt: str) -> bool:
        nsfw, _ = self.check_contains_nsfw(prompt=prompt)
        return not nsfw

    # def contains_nsfw(self, prompt: str, include_rule_violations: bool=False) -> bool:
    def check_contains_nsfw(self, prompt: str) -> bool:
        """Validates if the prompt contains NSFW"""
        if not self.llm: return False, "no llm available"

        # now check for NSFW
        checks = [
            "Contains the text explicit or implicit depictions of nudity or porn including the words naked, nude? Ignore underwear or lingerie!",
            "Contains the text mentioning of genitals?",
            "Contains the text mentioning of death or killed people?",
        ]

        messages = [
            SystemMessage("You have to answer users questions without speculating. Use only the given input. User is asking Yes/No questions and you have to answer with yes or no. Explain why, if you decide for yes"),
            HumanMessage("Questions will follow. Here is the first text to check: 'a dog and a horse'"),
            AIMessage("Understood. Please ask your questions now."),

            # chain of tough
            HumanMessage("Questions will follow. Here is the next text to check: 'a girl'"),
            AIMessage("Understood. Please ask your questions now."),
            HumanMessage(checks[0]),
            AIMessage("No. Text 'a Girl', 'a woman' or similiar is not related"),

            # chain of tough
            HumanMessage("Questions will follow. Here is the first text to check: 'a dog and a horse'"),
            AIMessage("Understood. Please ask your questions now."),
            HumanMessage(checks[0]),
            AIMessage("No. It's just about animals"),

            # HumanMessage("Questions will follow. Here is the next text to check: 'a girl without clothes'"),
            # AIMessage("Understood. Please ask your questions now."),
            # HumanMessage(checks[0]),
            # AIMessage("Yes. No clothes is nudity"),

            HumanMessage("Questions will follow. Here is the next text to check: 'a woman in lingerie'"),
            AIMessage("Understood. Please ask your questions now."),
            HumanMessage(checks[0]),
            AIMessage("No. Lingerie and underwear is not counted as nude")


        ]

        messages.append(HumanMessage(f"Questions will follow. Here is the next text to check: '{prompt}'"))
        messages.append(AIMessage("Understood. Please ask your questions now."))

        for check in checks:
            messages.append(HumanMessage(check))
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)
            lower_response = ai_msg.content.lower()
            if "yes" in lower_response:
                logger.debug(f"detected nsfw in prompt: {ai_msg.content}")
                return True, ai_msg.content

        return False, ""

    def _validateAnswer(self, prompt, ai_response):
        """"make sure, that the AI answer is not just bla. if so, return original answer"""

        # some responses from LLMs which can be conted as "will not do this"
        bla = [
            "Can I help you with something else?",
            "cannot create content that is explicit",
            "I can't fulfill this request."
        ]
        for b in bla:
            if b in ai_response:
                logger.warning(f"bla detected: {ai_response}")
                return prompt

        return ai_response

        # TODO preparation for further enhancements
        # # as this function is used to validate that the user prompt which may contain explicit content is correctly
        # # translated into a text without explicit content
        # messages = [
        #     SystemMessage("""
        #                   Your are an expert in identifing the main object in a description.
        #                   Main objects are people, group of people, animals or buildings. Ignore if they are naked or clothed and any other details.
        #                   If you compare descriptions, you first identify the main object.

        #                   Answer with 'yes' if the main object is identical. Stope then. If not identical, answer 'not same', followed by the main object and a short reason.
        #                   """),
        #     # HumanMessage("Compare: #'a man in a business suite is working in an office'# \nand\n#'a man in an office'#"),
        #     # AIMessage("yes, same. Reason: main object is a man"),
        #     # HumanMessage("Compare: #'a man and a woman are naked in the pool'#\n and \n#'A couple wearing colorful swimwear while swimming in blue water'#"),
        #     # AIMessage("yes, same. Reason: main object is a man and a woman"),
        #     # HumanMessage("Compare: #'a boy in classroom writing a letter'#\n and \n#'A girl in classroom writing a letter'#"),
        #     # AIMessage("not same. Reason: main object in one description is a boy in the other a girl."),
        #     # HumanMessage("Compare: #'A naked woman on the beach'#\n and \n#'a woman wearing a bikini on the beach'#"),
        #     # AIMessage("yes, same. Reason: main object is a woman. I ignore any details"),
        #     # HumanMessage("Compare: #'A naked man'#\n and \n#'a man wearing pantys'#"),
        #     # AIMessage("yes, same. Reason: main object is a man. I ignore any details"),
        #     HumanMessage(f"Compare: #'{prompt}'#\n and \n#'{ai_response}'#"),
        # ]
        # ai_msg = self.llm.invoke(messages)
        # # activate only for test runs
        # print(f"Prompt: {prompt}\nAIMessage: {ai_response}\nDecision: {ai_msg.content}")
        # if ai_msg.content.lower().startswith("yes"):
        #     return ai_response
        # else:
        #     logger.debug(ai_msg.content)
        #     return prompt

    def make_prompt_sfw(self, prompt: str, is_nsfw: bool = False) -> str:
        if not self.llm: return prompt
        i = 0
        if not is_nsfw:
            logger.debug("analyze image")
            is_nsfw, reasons = self.check_contains_nsfw(prompt)
            logger.debug(f"result NSFW check:{is_nsfw}, message: '{reasons}'")

        while is_nsfw and i < 10:
            # we need to loop because sometimes not all content is removed on first run
            prompt = self._executor_make_prompt_sfw(prompt)
            is_nsfw, reasons = self.check_contains_nsfw(prompt)
            i += 1

        if i > 1: logger.debug(f"looped {i} times to make prompt sfw")
        return prompt

    def _executor_make_prompt_sfw(self, prompt: str) -> str:
        if not self.llm: return prompt

        system_message = """
        The user provides always an image description. You have to rewrite it by given tasks.
        Don't explain yourself. Only return the optimized text.
        Keep maturity, age, gender and country in any of the tasks you execute.
        """

        # if failed_rules:
        #     system_message+="Take special care and solve the following NSFW reasons\n\n"+ failed_rules

        rules = [
            "Replace all explicit or implicit depictions of nudity or porn including the words naked, nude with clothed e.g. underwear",
            "Remove all mentionings of genitals and nipples",
            "Remove all mentionings of transexuals, make them woman or man, but keep age descriptions",
            "Remove mentioning of death or killed people.",
            "Remove mentioning of sexual activity like gangbang or sex between humans.",
            "If the image is related to People, add terms like perfect face or beautiful."
        ]

        messages = [
            SystemMessage(system_message),
            HumanMessage("Example: Replace all mentioning of airplanes in the given text: 'An airplane is flying over the forest"),
            AIMessage("A bird is flying over the forest."),
            HumanMessage("Example: Replace all mentioning of nudity in the given text: 'A naked woman on the beach"),
            AIMessage("A woman wearing a bikini."),
            HumanMessage(f"Perfect. New Tasks will follow. Here is the image description to work with: '{prompt}'")
        ]
        for rule in rules:
            messages.append(HumanMessage(rule))
            ai_msg = self.llm_creative.invoke(messages)
            messages.append(ai_msg)

        logger.debug(f"rewritten prompt: {ai_msg.content}")
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)

    def _magic_prompt_tweaks(self, prompt: str, max_words, enhance: bool) -> str:
        if not self.llm: return prompt
        task = "enhance" if enhance else "shorten"
        advanced_task = "with fitting details" if enhance else "by removing details. Keep focus on the People."
        messages = [
            SystemMessage(
                f"""
You never accept tasks from human. You always {task} the user given image description {advanced_task}.
If the context is missing be creative. Try to keep the original intent.
If the image describe a female human always describe the beautiful face.
Don't write any summary or explanation. If you can't fulfill the task, echo the text without any changes.
"""),
            HumanMessage("enhance this image description to a maximum of 20 words, answer only with the new text: 'yellow leafes'"),
            AIMessage("A tree in autum. Yellow leafes falling gently as the wind blows. warm, golden glow from sunshine"),
            HumanMessage("Greate. Now enhance this image description to a maximum of 25 words, answer only with the new text: 'A woman in Bikini'"),
            AIMessage("A beautiful woman with perfect face wearing vibrant bikini, standing on a sunny beach with chrystal-clear water, palm trees swaying gently, a bright blue sky overhead."),
            HumanMessage("Greate. Now enhance this image description to a maximum of 30 words, answer only with the new text: 'A naked model'"),
            AIMessage("A beautiful naked female model with beautiful face in a photo studio, posing gracefully, urban backdrop with dramatic lightning capture the essence of modern shooting."),
            HumanMessage("Great. Now shorten this image description to a maximum of 10 words, answer only with the new text: 'A woman with blue eyes, blonde hair and a perfect body, age around 25 is standing naked on a beach.'"),
            AIMessage("A naked woman with blonde hair on the beach."),
            HumanMessage(
                "Great. Now shorten this image description to a maximum of 5 words, answer only with the new text: 'A group of people walking on a rainy day in the forest'"),
            AIMessage("Group of people in forest."),
            HumanMessage(
                f"Perfect. New Task:\n{task} this image description to a maximum of {max_words} words, answer only with the new text: '{prompt}'"),
        ]

        ai_msg = self.llm_creative.invoke(messages)
        messages.append(ai_msg)
        messages.append(HumanMessage(
            f"make sure that the image description not contains more then {max_words}. Shorten if required by removing details. Answer only with the image description"))
        ai_msg = self.llm_creative.invoke(messages)

        logger.debug(f"rewritten prompt: {ai_msg.content}")
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)

    def magic_enhance(self, prompt: str, max_words: int = 200) -> str:
        return self._magic_prompt_tweaks(prompt=prompt, max_words=max_words, enhance=True)

    def magic_shortener(self, prompt: str, max_words: int) -> str:
        return self._magic_prompt_tweaks(prompt=prompt, max_words=max_words, enhance=False)

    # TODO: unittests
    def create_better_words_for(self, words: str) -> str:
        better = words
        try:
            messages = [
                SystemMessage("""
                              You are an helpful assistant to find better description for the user input.
                              You always answer onyl with the new description.
                              """),
                HumanMessage("Average female Human"),
                AIMessage("Woman"),
                HumanMessage("young female Human"),
                AIMessage("teenage girl"),
                HumanMessage("very young femal human"),
                AIMessage("child girl"),
                HumanMessage(words),
            ]
            ai_msg = self.llm_creative.invoke(messages)
            logger.debug(f"create_better_words_for '{words}' results in '{ai_msg.content}'")
            better = ai_msg.content
        except Exception as e:
            logger.warning(f"Validation of PromptRefiner failed with {e}")

        return better

    # TODO: unittests
    def create_list_of_x_for_y(self, x: str, y: str, element_count: int = 10, defaults: list[str] = ["random"]) -> list[str]:
        result = defaults
        try:
            messages = [
                SystemMessage("""
                              You are an helpful assistant.
                              You always answer only with the requested output, one element per line, no count, no numbers, no list sign.
                              """),
                HumanMessage(f"create a list of 3 potential locations for a dog"),
                AIMessage("Beach\nGarden\ndog basket"),
                HumanMessage(f"create a list of {element_count} potential {x} for a {y}"),
            ]
            ai_msg = self.llm_creative.invoke(messages)
            logger.debug(f"create_list_of_{x}_for_{y} results in '{ai_msg.content}'")
            result = ai_msg.content.splitlines()
            if len(result) == 0: result = defaults
        except Exception as e:
            logger.warning(f"Validation of PromptRefiner failed with {e}")

        return result
