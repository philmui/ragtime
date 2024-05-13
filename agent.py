import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("agent")

from llama_index.llms.openai import OpenAI

from globals import (
    gpt3_model,
    gpt4_model
)

from llama_index.core.prompts import PromptTemplate
SIMPLE_TEMPLATE = (
    "You are a friendly and helpful agent. "
    "Please respond to to user's query:\n"
    "Query: {query_str}\n"
    "Response: "
)
simple_template = PromptTemplate(SIMPLE_TEMPLATE)

RAG_TEMPLATE = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

qa_template = PromptTemplate(RAG_TEMPLATE)


class RAGTimeAgent:

    llm: OpenAI = None

    def __init__(
        self,
        llm: OpenAI,
        verbose = True
    ):
        self.llm = llm
        self.verbose = verbose


    def complete(self, query_str: str) -> str:

        response = self.llm.complete(
            simple_template.format(query_str=query_str)
        )

        return response.text


class AgentFactory:
    _agent: RAGTimeAgent = None

    @classmethod
    def get_agent(cls) -> RAGTimeAgent:
        if cls._agent == None:
            cls._agent = RAGTimeAgent(
                llm=gpt4_model,
            )
        return cls._agent