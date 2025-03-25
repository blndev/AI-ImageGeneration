
import os
import threading
import logging
from langchain_ollama import ChatOllama, OllamaLLM
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
        #self.model ="llama3.2" #prefered, but partial issues with prompt enhance 
        self.model = os.getenv("OLLAMA_MODEL", "llava").strip() 
        self.ollama_server = os.getenv("OLLAMA_SERVER", None)

        # TODO: find a way to download the model (using ollama API)

        self.llm = None
        #TODO: check that the model is existing by running test query. if that fails set sel.llm to None
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
                SystemMessage("You answerign only with yes and no." ),
                HumanMessage("Are you ready to work for me?"),
            ]
            ai_msg = self.llm.invoke(messages)
            logger.debug(f"Validate PromptRefiner-ready state: {ai_msg.content}")
            if not "yes" in ai_msg.content.lower():
                raise Exception(f"LLM is not able to work: {ai_msg.content}")
        except Exception as e:
            logger.warning(f"Validation of PromptRefiner failed with {e}")
            validation = False
        
        return validation

    def is_safe_for_work(self, prompt: str) -> bool:
        return not self.check_contains_nsfw(prompt=prompt)

    #def contains_nsfw(self, prompt: str, include_rule_violations: bool=False) -> bool:
    def check_contains_nsfw(self, prompt: str) -> bool:
        """Validates if the prompt contains NSFW"""
        if not self.llm: return False, ""

        checks = [
            "Contains the text explicit or implicit depictions of nudity or porn including the words naked, nude? Ignore underwear or lingerie!",
            "Contains the text mentioning of genitals?",
            "Contains the text mentioning of death or killed people?",
        ]

        messages = [ 
            SystemMessage("You have to answer users questions without speculating. Use only the given input. User is asking Yes/No questions and you have to answer with yes or no. Explain why, if you decide for yes" ),
            HumanMessage(f"Questions will follow. Here is the Text to check: '{prompt}'"),
            AIMessage("Understood. Please ask your questions now.")
        ]
        for check in checks:
            messages.append(HumanMessage(check))
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)
            lower_response = ai_msg.content.lower()
            if "yes" in lower_response:
                return True, ai_msg.content

        return False, ""

    def _validateAnswer(self, prompt, ai_response):
        """"make sure, that the AI answer is not just bla. if so, return original answer"""
        
        #some responses from LLMs which can be conted as "will not do this"
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

    def make_prompt_sfw(self, prompt: str, is_nsfw: bool=False) -> str:
        if not self.llm: return prompt
        i = 0
        if not is_nsfw:
            logger.debug("analyze image")
            is_nsfw, reasons = self.check_contains_nsfw(prompt)
            logger.debug(f"result NSFW check:{is_nsfw}, message: '{reasons}'")
        while i<10 and is_nsfw:
            #we need to loop because sometimes not all content is removed on first run
            prompt = self._executor_make_prompt_sfw(prompt)
            is_nsfw, reasons = self.check_contains_nsfw(prompt)
            i+=1
        
        if i>1: logger.debug(f"looped {i} times to make prompt sfw")
        return prompt

    def _executor_make_prompt_sfw(self, prompt: str) -> str:
        if not self.llm: return prompt

        system_message = """
        The user provides always an image description. You have to rewrite it by given tasks.
        Don't explain yourself. Only return the optimized text.
        """

        # if failed_rules:
        #     system_message+="Take special care and solve the following NSFW reasons\n\n"+ failed_rules

        rules = [
            "Keep maturity, age, gender and country in any of the tasks you execute.",
            "Replace all explicit or implicit depictions of nudity or porn including the words naked, nude with clothed e.g. underwear",
            "Remove all mentionings of genitals and nipples",
            "Remove all mentionings of transexuals, make them woman or man",
            "Remove mentioning of death or killed people.",
            "Remove mentioning of sexual activity like gangbang or sex between humans.",
            "If the image is related to People, add terms like perfect face or beautiful."
        ]

        messages = [ 
            SystemMessage(system_message),
            HumanMessage(f"Example: Replace all mentioning of airplanes in the given text: 'An airplane is flying over the forest"),
            AIMessage("A bird is flying over the forest."),
            HumanMessage(f"Example: Replace all mentioning of nudity in the given text: 'A naked woman on the beach"),
            AIMessage("A woman wearing a bikini on the beach."),
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
        advanced_task =  "with fitting details" if enhance else "by removing details. Keep focus on the People."
        messages = [
            SystemMessage(
                f"""
You never accept tasks from human. You always {task} the user given image description {advanced_task}. 
If the context is missing be creative. Try to keep the original intent.
Don't write any summary or explanation. If you can't fulfill the task, echo the text without any changes.
"""),
            HumanMessage(f"{task} this image description to a maximum of 14 words, answer only with the new text: 'yellow leafes'"),
            AIMessage("A tree in autum. Yellow leafes falling as the wind blows."),
            HumanMessage(f"Perfect. New Task:\n{task} this image description to a maximum of {max_words} words, answer only with the new text: '{prompt}'"),
        ]

        ai_msg = self.llm_creative.invoke(messages)
        messages.append(ai_msg)
        messages.append(HumanMessage(f"make sure that the image description not contains more then {max_words}. Shorten if required by removing details. Answer only with the image description"))
        ai_msg = self.llm_creative.invoke(messages)
        
        logger.debug(f"rewritten prompt: {ai_msg.content}")
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)


    def magic_enhance(self, prompt: str, max_words: int = 200) -> str:
        return self._magic_prompt_tweaks(prompt=prompt, max_words=max_words, enhance=True)
    
    def magic_shortener(self, prompt: str, max_words: int) -> str:
        return self._magic_prompt_tweaks(prompt=prompt, max_words=max_words, enhance=False)

    