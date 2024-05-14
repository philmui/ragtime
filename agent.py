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
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
    VectorIndexRetriever
)
from llama_index.core.llms import (
    ChatMessage,
    MessageRole
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine
)
from llama_index.core.response_synthesizers import ResponseMode
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
from llama_index.core import get_response_synthesizer, load_index_from_storage

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
    embedding_model,
    gpt3_model,
    gpt4_model
)

Settings.llm = gpt3_model
Settings.embed_model = embedding_model

SYSTEM_TEMPLATE = (
    ""
)

SIMPLE_TEMPLATE = (
    "You are a friendly and helpful agent. "
    "Please respond to to user's query:\n"
    "Query: {query_str}\n"
    "Response: "
)

RAG_TEMPLATE = (
    "Respond to the user's Query given the Context information below. \n"
    "---------------------\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "If there is not sufficient information in the Context to answer "
    "the user's Query, please reply 'I don't have sufficient verifiable information.'\n"
    "Query: {query_str}\n"
    "Response: "
)

REFINE_RAG_TEMPLATE = (
    "We have the opportunity to refine our response to the original answer "
    "(only if needed) with some more context below.\n"
    "---------------------\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "Given the new context, refine the original answer to better answer the Query: \n"
    "Query: {query_str}. \n"
    "If the context isn't useful, output the original response again.\n"
    "Original Response: {existing_response}\n"
    "New Response: "
)

simple_template = PromptTemplate(SIMPLE_TEMPLATE)
rag_template = PromptTemplate(RAG_TEMPLATE)
refine_template = PromptTemplate(REFINE_RAG_TEMPLATE)

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
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=6,
        vector_store_query_mode="mmr"
    )
    query_engine = vector_index.as_query_engine(
        similarity_top_k=6,
        vector_store_query_mode="mmr",
        vector_store_kwargs={
            "mmr_prefetch_k": 20,
        },
        response_synthesizer=get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            refine_template=refine_template
        )
    )

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
            use_async=True,
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=32),
                embedding_model
            ]
        )
        # RAGTimeAgent.vector_index.refresh_ref_docs(documents=documents)
        # RAGTimeAgent.vector_index.build_index_from_nodes(nodes=nodes)

        return len(documents)

    async def aquery(self, query_str: str) -> str:
        output = ""
        if RAGTimeAgent.query_engine is not None:
            response = await RAGTimeAgent.query_engine.aquery(
                query_str
            )
            output = response.response
        else:
            output = await self.acomplete(query_str)
        return output


    async def acomplete(self, query_str: str) -> str:

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