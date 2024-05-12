from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from llama_index.llms.openai import (
    OpenAI
)
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingModeModel
)

embedding_model = OpenAIEmbedding(
    model=OpenAIEmbeddingModeModel.TEXT_EMBED_3_SMALL
)

GPT3_MODEL = "gpt-3.5-turbo-0125"
GPT4_MODEL = "gpt-4-turbo-preview"
GPT4_0125_MODEL = "gpt-4-0125-preview"

# LlamaIndex
gpt3_model = OpenAI(model=GPT3_MODEL)
gpt4_model = OpenAI(model=GPT4_MODEL)

