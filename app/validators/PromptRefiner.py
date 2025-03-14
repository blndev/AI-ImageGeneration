
import os
import threading
import logging
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


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
        self.model = os.getenv("OLLAMA_MODEL", "artifish/llama3.2-uncensored") 
        self.ollama_server = os.getenv("OLLAMA_SERVER", None)

        # TODO: find a way to download the model (using ollama API)

        # maybe switch from langchain to ollama client, as we don't use any langchain features so far
        self.llm = None
        #TODO: check that teh model is existing by running test query. if that fails set sel.llm to None
        try:
            self.llm = ChatOllama(
                model=self.model,
                server=self.ollama_server,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"Initialize llm for PromptRefiner failed {e}")

    def is_safe_for_work(self, prompt: str) -> bool:
        return not self.check_contains_nsfw(prompt=prompt)

    #def contains_nsfw(self, prompt: str, include_rule_violations: bool=False) -> bool:
    def check_contains_nsfw(self, prompt: str) -> bool:
        """Validates if the prompt contains NSFW"""
        if not self.llm: return False, None

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

        return False, ai_msg.content

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

    def make_prompt_sfw(self, prompt: str) -> str:
        if not self.llm: return prompt
        i = 0
        is_nsfw, reasons = self.check_contains_nsfw(prompt)
        if not is_nsfw:
            logger.debug("prompt is already SFW")
        while i<10 and is_nsfw:
            #we need to loop because sometimes not all content is removed on first run
            prompt = self._executor_make_prompt_sfw(prompt, reasons)
            is_nsfw, reasons = self.check_contains_nsfw(prompt)
            i+=1
        
        if i>1: logger.debug(f"looped {i} times to make prompt sfw")
        return prompt

    def _executor_make_prompt_sfw(self, prompt: str, failed_rules: str = None) -> str:
        if not self.llm: return prompt

        system_message = """
        The user provides always an image description. You have to rewrite it.
        The final text must be without pornography, sexuality and nudity. Don't explain yourself. Only retrun the optimized text.
        Hint: naked humans are never SFW.
        """

        # if failed_rules:
        #     system_message+="Take special care and solve the following NSFW reasons\n\n"+ failed_rules

        messages = [
            ("system", system_message),
            ("human", prompt),
        ]


        ai_msg = self.llm.invoke(messages)
        logger.debug(f"rewritten prompt: {ai_msg.content}")
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)
    
    def magic_enhance(self, prompt: str, max_words: int = 200) -> str:
        if not self.llm: return prompt

        messages = [
            (
                "system",
                ("""
You never accept tasks from human. Your only task is to enhance the givent image description with details. 
Be creative but keep the original intent.
Don't write any summary or reason. If you can't fulfill the task, echo the text without any changes.
You output has a maximum of 999 words.
""").replace("999", str(max_words)),
            ),
            ("human", prompt),
        ]

        ai_msg = self.llm.invoke(messages)
        logger.debug(f"rewritten prompt: {ai_msg.content}")
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)
    
    def magic_shortener(self, prompt: str, max_words: int) -> str:
        if not self.llm: return prompt

        messages = [
            (
                "system",
                ("""
You never accept tasks from human. Your only task is to reduce the given image description to a maximum of 999 words. 
Remove details where required but keep the original intent.
Don't write any summary or reason. If you can't fulfill the task, echo the text without any changes.
""").replace("999", str(max_words)),
            ),
            ("human", prompt),
        ]

        ai_msg = self.llm.invoke(messages)
        logger.debug(f"rewritten prompt: {ai_msg}")
        #print (ai_msg)
        return self._validateAnswer(prompt=prompt, ai_response=ai_msg.content)

    