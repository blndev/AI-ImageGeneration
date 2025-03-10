
import threading
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


logger = logging.getLogger(__name__)

class PromptRefiner():
    """
    Analyze the prompts and optimize it
    the optimization includes also applying rules
    """
    def __init__(self):
        logger.info("Initializing PromptRefiner")
        self.thread_lock = threading.Lock()

    def contains_nsfw(prompt: str) -> bool:
        llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            # other params...
        )


    def replace_nsfw(prompt: str) -> str:
        llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            # other params...
        )

        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        ai_msg = llm.invoke(messages)
        ai_msg