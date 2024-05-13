import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger("agent")
_logger.addHandler(logging.StreamHandler(stream=sys.stdout))
from pprint import pprint

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import (
    OpenAIEmbedding, 
    OpenAIEmbeddingModeModel
)
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import (
    FireCrawlWebReader,
    SimpleWebPageReader,
    SpiderWebReader
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

import os
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SPIDER_API_KEY = os.getenv("SPIDER_API_KEY")

QDRANT_URL = "https://94dd78f2-8663-4bdf-80e0-db8f174f619b.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "ragtime"

from typing import List, Sequence

from markdown import MDConverterFactory

from globals import (
    gpt3_model,
    gpt4_model
)

Settings.llm = gpt4_model
Settings.embed_model = OpenAIEmbedding(
    model=OpenAIEmbeddingModeModel.TEXT_EMBED_3_LARGE
)

SIMPLE_TEMPLATE = (
    "You are a friendly and helpful agent. "
    "Please respond to to user's query:\n"
    "Query: {query_str}\n"
    "Response: "
)
simple_template = PromptTemplate(SIMPLE_TEMPLATE)

RAG_TEMPLATE = (
    "Respond to the user's Query given the Context information below. \n"
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "If there is not sufficient information in the Context to answer "
    "the user's Query, please reply 'I don't have sufficient verifiable information.':\n"
    "Query: {query_str}\n"
)

qa_template = PromptTemplate(RAG_TEMPLATE)

class RAGTimeAgent:

    q_client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    q_aclient = qdrant_client.AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    vector_store = QdrantVectorStore(
        collection_name=QDRANT_COLLECTION_NAME,
        aclient=q_aclient,
        prefer_grpc=True,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index: VectorStoreIndex = None

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
        verbose = True
    ):
        self._llm = llm
        self.verbose = verbose


    def ingest_urls(
        self,
        urls: List[str]
    ) -> int:
        
        if urls is None or len(urls) == 0 or urls[0] == None:
            return 0
        
        # documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        documents: Sequence[Document] = []
        for url in urls:
            docs = RAGTimeAgent.spider_reader.load_data(url=url)
            for d in docs:
                d.text = MDConverterFactory.get_converter().convert(d.text)
            documents.extend(docs)
        
        _logger.info(f"crawled {len(documents)} pages")
        
        RAGTimeAgent.vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=RAGTimeAgent.storage_context,
            use_async=True
        )
        self._query_engine = RAGTimeAgent.vector_index.as_query_engine(use_async=True)

        # index = SummaryIndex.from_documents(
        #     documents,
        #     show_progress=True
        # )
        # self._query_engine = index.as_query_engine()

        return len(documents)

    async def aquery(self, query_str: str) -> str:
        output = ""
        if self._query_engine is not None:
            response = await self._query_engine.aquery(query_str)
            output = response.response
        else:
            output = await self.complete(query_str)
        return output


    async def complete(self, query_str: str) -> str:

        response = await self.llm.acomplete(
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