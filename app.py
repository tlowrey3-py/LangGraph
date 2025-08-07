import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType, VectorizableTextQuery

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version="2024-02-15-preview",
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list

graph_builder = StateGraph(State)

def chatbot(state: State):
    # Get the retrieved documents from the state
    retrieved_documents = state.get("documents", [])
    
    # Format the documents into a single context string
    context_str = "Relevant contract information:\n\n"
    if not retrieved_documents:
        context_str = "No relevant documents found.\n"
    else:
        for i, doc in enumerate(retrieved_documents):
            context_str += f"--- Document {i+1} ---\n"
            context_str += f"Title: {doc.get('title', 'N/A')}\n"
            context_str += f"Abbreviated Title: {doc.get('abbreviated_title', 'N/A')}\n"
            context_str += f"Implementation Date: {doc.get('implementation_date', 'N/A')}\n"
            context_str += f"Content: {doc.get('content', 'N/A')}\n\n"

    # The main system prompt with instructions
    system_prompt = (
        """You are a professional chatbot designed to assist pilots with contract-related questions.

                Context:
                - You are working with the Southwest Airlines pilot contract (CBA) and, at times, the Southwest Airlines pilot FAQ.
                - FAQ entries are helpful but **not part of the official contract** and must be cited as such.
                - You will be provided the current date, the user’s question, and relevant context sections.
                - Each context section includes: `title`, `abbreviated_title`, `content`, and `content_type`.

                Instructions:
                - Answer the user’s question using only the information provided in the context.
                - If the question does not have enough information ask the user for more details.
                - Never assume or infer meaning not found in the context.
                - If a term or acronym is not defined in the provided context, explicitly state that it is not found.
                - Do not interpret or guess the meaning of acronyms unless the context explicitly defines them.
                - Only use direct quotes when they clearly answer the question. Quote verbatim and use quotation marks.
                - Use only the exact terminology from the context—never substitute or paraphrase unfamiliar terms.
                - If a portion of the answer is based on an FAQ, clearly indicate that it comes from the FAQ.

                Citations:
                - Always cite the `abbreviated_title(s)` used to answer the question.
                - End your response with a new line followed by the cited `abbreviated_title(s)`.

                Response Style:
                - Be brief and get to the point quickly. Pilots prefer clear, no-nonsense answers.
                - If a direct answer is possible from the context, give it first. Add clarification only if it's essential to understanding.
                - Avoid repeating information or offering extra detail that is not directly relevant to the user's question.
                - Do not summarize the entire context—respond only with what’s needed to answer the question accurately.
                - Maintain a professional and respectful tone while prioritizing clarity and efficiency."""
    )

    # Combine the user's question with the context
    last_message = state["messages"][-1]
    if isinstance(last_message, dict):
        user_question = last_message.get("content", "")
    else:
        user_question = getattr(last_message, "content", "")

    prompt_with_context = (
        f"Context from retrieved documents:\n{context_str}\n"
        f"Based on the context above, please answer the following question:\n"
        f"Question: {user_question}"
    )

    # Create the final messages list for the LLM
    final_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_with_context},
    ]

    # Call the language model
    response = llm.invoke(final_messages)
    
    # Return the response and the documents to be added to the state
    return {"messages": [response], "documents": retrieved_documents}



def azure_search_node(state: State):
    # The last message in the state is the user's query.
    # Note: `add_messages` converts the input dict to a message object (e.g., HumanMessage).
    last_message = state["messages"][-1]

    # Ensure the last message is from the user before searching.
    # In this graph, it always will be, but this is a good safeguard.
    if last_message.type != "human":
        return {} # Return nothing to update the state

    query = last_message.content
    
    # Set up Azure Search client
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    index_name = os.environ["AZURE_SEARCH_INDEX"]
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=10,
        fields="contentVector",
        exhaustive=True
    )
    
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content_type", "title", "content", "abbreviated_title", "implementation_date"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=10
    )
    
    contract_chunks = []
    for result in results:
        title = result.get("title", "")
        abbr = result.get("abbreviated_title", "")
        date = result.get("implementation_date", "")
        content = result.get("content", "")
        contract_chunks.append({
            "title": title,
            "abbreviated_title": abbr,
            "implementation_date": date,
            "content": content
        })
    return {"documents": contract_chunks[:5]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("azure_search", azure_search_node)

graph_builder.add_edge(START, "azure_search")
graph_builder.add_edge("azure_search", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str, messages: list):
    # Add user message to the history
    messages.append({"role": "user", "content": user_input})
    
    # The initial state for the graph includes the full message history
    initial_state = {"messages": messages}
    
    # Keep track of the final message history
    final_messages = messages
    
    # Stream events from the graph
    for event in graph.stream(initial_state):
        # The 'chatbot' node is the one that produces the assistant's message
        if "chatbot" in event:
            assistant_message = event["chatbot"]["messages"][-1]
            print("Assistant:", assistant_message.content)
        
        # Update the message history with the final state from the graph
        if END in event:
            final_messages = event[END].get("messages", messages)
            
    return final_messages

messages = []
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        messages = stream_graph_updates(user_input, messages)
    except Exception as e:
        print(f"An error occurred: {e}")
        break
