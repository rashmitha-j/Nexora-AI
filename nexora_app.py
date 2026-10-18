import asyncio
import os

import chainlit as cl
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your local .env file."
    )

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.8-flash"

@cl.on_chat_start
async def start():
    await cl.Message(
        content="# Welcome to Nexora AI\n\n"
        "Your AI assistant, powered by Google Gemini. "
        "Ask me a question to get started."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=message.content,
        )
        answer = response.text or "Gemini returned an empty response."
        await cl.Message(content=answer).send()
    except Exception:
        await cl.Message(
            content="I couldn't get a response. Check your API key, "
            "internet connection, and Gemini API access."
        ).send()
