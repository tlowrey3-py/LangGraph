import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version="2024-02-15-preview",
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    # Append the assistant's response to the conversation history
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [{"role": "assistant", "content": response.content}]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str, messages: list):
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    # Stream graph updates with full history
    for event in graph.stream({"messages": messages}):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])
            # Update messages with the latest state
            messages = value["messages"]
    return messages

messages = []
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        messages = stream_graph_updates(user_input, messages)
    except Exception:
        # fallback if input() is not available
        # user_input = "What do you know about LangGraph?"
        print("An error occurred. Please try again.")
        break
