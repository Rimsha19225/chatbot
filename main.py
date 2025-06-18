import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

provider = AsyncOpenAI(
        api_key = openai_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-1.5-flash-latest",
    openai_client= provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent1 = Agent(
    instructions= "You are a helpful assistant.",
    name= "assistant"
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am your helper as agent. How can i help now?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])

    msg = cl.Message(content="")

    history.append({"role" : "user", "content" : message.content})
    result = Runner.run_streamed(
        agent1,
        input = history,
        run_config= run_config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent) and (token := event.data.delta):
            await msg.stream_token(token)

        await msg.send()


    history.append({"role" : "assistant", "content" : result.final_output})
    cl.user_session.set("history", history)
    # await cl.Message(content=result.final_output).send()
