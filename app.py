import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("app")

import chainlit as cl
from globals import (
    gpt3_model,
    gpt4_model
)

from agent import AgentFactory
from utils import extract_url

@cl.on_message
async def chat(message: cl.Message):
    response = ""
    try:
        url = extract_url(message.content)
        agent = AgentFactory.get_agent([url])
        if url is not None:
            response = "What can I answer for you?"
        else:
            response = agent.query(message.content)
    except Exception as e:
        _logger.error(f"agent error: {e}")        

    # Send a response back to the user
    await cl.Message(
        content=f"{response}"
    ).send()


@cl.on_chat_start
async def start():

    await cl.Message(
        content="Let me know a URL!"
    ).send()