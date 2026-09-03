from langgraph.graph import END, StateGraph, add_messages
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from typing import Annotated, List, Optional
from langchain_openai import AzureChatOpenAI
from langfuse import Langfuse


# QDRANT_URL = os.getenv("QDRANT_URL")
# QdrantClient_instance = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))


load_dotenv()
langfuse = Langfuse()

langfuse_callback_handler = CallbackHandler()
# emmbed_model_instance = SentenceTransformer('all-MiniLM-L6-v2')

def get_azure_chat_openai_client():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name="gpt-4.1-mini",
        api_version="2024-12-01-preview",
        temperature="0.3",
    )
# def get_faq_from_qdrant(
#     vector: List[float], 
#     top_k: int, 
#     collection_name: Optional[str] = None
# ):
#     print('QDRANT----URL---', QDRANT_URL)
#     """
#     Fetch FAQ documents from Qdrant based on the provided vector.
#     """
#     if not collection_name:
#         raise ValueError("collection_name must be provided")
#     if not vector:
#         raise ValueError("vector must be provided and non-empty")
#     try:
#         faq_docs = QdrantClient_instance.query_points(
#             collection_name=collection_name,
#             query=vector,
#             limit=top_k
#         )
#         return faq_docs
#     except Exception as e:
#         print(f"Error fetching FAQ docs from Qdrant: {e}")
#         return None
    
# def convert_user_query_to_vector(user_query):
#     """
#     Convert a user query to a vector using a preloaded SentenceTransformer model.
#     """
#     vector = emmbed_model_instance.encode(user_query).tolist()
#     return vector

class ChatAgentState:
    messages: Annotated[List[dict], add_messages]
    docs: list[dict]
    
    
def chat_node(state: ChatAgentState):
    # Implement the logic for the chat node here
    print("Invoking Azure Chat OpenAI client...")
    prompt = langfuse.get_prompt("medical_assistant")
    print("Prompt:", prompt)
    messages = state["messages"]
    print("Messages:", messages)
    # vector_quer_query = convert_user_query_to_vector(messages[-1].content)
    # rag_chunks = get_faq_from_qdrant(vector_quer_query, top_k=5, collection_name="faq_collection")
    # print("Retrieved FAQ docs:", rag_chunks)
    docs_data = []
    # if rag_chunks and rag_chunks.points:
    #     for doc in rag_chunks.points:
    #         docs_data.append({
    #             "text": doc.payload.get("text", ""),
    #             "score": doc.score
    #         })
    full_prompt = prompt.compile(
    user_query=messages[-1].content,
    docs=[]
)
    print("Full prompt:", full_prompt)
    llm = get_azure_chat_openai_client()
    llm_response = llm.invoke(full_prompt)
    print("LLM response:", llm_response)
    state["messages"].append({"role": "assistant", "content": llm_response.content})
    return state
    


def chat_agent_graph():
    """Build a small sample graph similar to sema_agent, but with fewer nodes."""
    graph = StateGraph(ChatAgentState)

    graph.add_node("start", chat_node)

    graph.set_entry_point("start")
    graph.add_edge("start", END)

    return graph.compile().with_config({"callbacks": [langfuse_callback_handler]})


# Exported graph object
chat_agent = chat_agent_graph()