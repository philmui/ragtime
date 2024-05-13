import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("app")

import chainlit as cl
from globals import (
    gpt3_model,
    gpt4_model
)

from agent import AgentFactory

@cl.on_message
async def main(message: cl.Message):
    response = ""
    try:
        agent = AgentFactory.get_agent()
        response = agent.complete(message.content)
    except Exception as e:
        _logger.error(f"agent error: {e}")        

    # Send a response back to the user
    await cl.Message(
        content=f"{response}"
    ).send()


@cl.on_chat_start
async def start():

    await cl.Message(
        content="Hello there!"
    ).send()