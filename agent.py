import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger("agent")
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))
from pprint import pprint

from llama_index.llms.openai import OpenAI
from llama_index.core.llms.function_calling import FunctionCallingLLM

from llama_index.core import (
    Document,
    SummaryIndex
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.readers.web import (
    FireCrawlWebReader,
    SimpleWebPageReader,
    SpiderWebReader
)

import os
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SPIDER_API_KEY = os.getenv("SPIDER_API_KEY")

from typing import List, Sequence

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

    firecrawl_reader = FireCrawlWebReader(
        api_key=FIRECRAWL_API_KEY,
        mode="scrape",
        # params={
        #     'crawlerOptions': {
        #         'excludes': ['blog/*'],
        #         'includes': [], # leave empty for all pages
        #         'limit': 5,
        #     },
        #     'pageOptions': {
        #         'onlyMainContent': True
        #     }
        # }
    )

    spider_reader = SpiderWebReader(
        api_key=SPIDER_API_KEY,
        mode="scrape",
    )

    @property
    def llm(self) -> FunctionCallingLLM:
        return self._llm

    @property
    def url(self) -> str:
        return self._url
    
    @url.setter
    def urls(self, url_list: List[str]):
        if url_list is None:
            raise ValueError("URLs cannot be None.")
        self._urls = url_list

    @property
    def query_engine(self) -> BaseQueryEngine:
        return self._query_engine

    def __init__(
        self,
        llm: FunctionCallingLLM,
        urls: List[str],
        verbose = True
    ):
        self._llm = llm
        self._urls = urls
        self.verbose = verbose

        if urls is not None and urls[0] is not None:
            self._ingest_urls(urls=urls)


    def _ingest_urls(
        self,
        urls: List[str]
    ) -> int:
        
        # documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        documents: Sequence[Document] = []
        for url in urls:
            docs = RAGTimeAgent.spider_reader.load_data(url=url)
            documents.extend(docs)
        
        _logger.info(f"crawled {len(documents)} pages")
        index = SummaryIndex.from_documents(
            documents,
            show_progress=True
        )
        self._query_engine = index.as_query_engine()

        return len(documents)

    def query(self, query_str: str) -> str:
        output = ""
        if self._query_engine is not None:
            response = self._query_engine.query(query_str)
            output = response.response
        else:
            output = self.complete(query_str)
        return output

    def complete(self, query_str: str) -> str:

        response = self.llm.complete(
            simple_template.format(query_str=query_str)
        )

        return response.text


class AgentFactory:
    _agent: RAGTimeAgent = None

    @classmethod
    def get_agent(cls, urls: List[str]) -> RAGTimeAgent:
        if cls._agent == None:
            cls._agent = RAGTimeAgent(
                llm=gpt4_model,
                urls=urls
            )
        return cls._agent