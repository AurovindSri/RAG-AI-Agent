from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import os
import langchain_core.chat_history as lc
from langchain_core.messages.base import message_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import List
import uvicorn
from datetime import datetime, timezone
from langchain_core.tools import Tool
import boto3
import json
import base64
from fastapi.responses import JSONResponse
import traceback

# Vector store setup
# from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


"""embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    "vector_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  
)"""

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333) 

# Load vector store from Qdrant
vector_store = Qdrant(
    client=client,
    collection_name="test_collection",
    embeddings=embedding_model,
)

def retrieve_db(query: str, threshold: float = 0.0):    #Added similarity threshold to avoid irrelevant context for prompts if not needed
    """Retrieve information related to a query from vector Database."""
    if not vector_store:
        return ""
    
    # Get documents with similarity scores
    #results = vector_store.similarity_search_with_score(query, k=2)                    #Good enough but might give redundant context
    results = vector_store.max_marginal_relevance_search(query, k=3, lambda_mult=threshold)   #Avoids giving redundant context & gives more context
    
    # Filter based on threshold
    filtered_docs = results
    
    # [
    #     doc for doc, score in results if score >= threshold
    # ]
    
    # if not filtered_docs:
        # return ""  # No relevant context

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in results
    )
    return serialized

# Read API keys from AWS Secrets Manager
def get_secret():
    secret_name = "llm-api-key"
    region_name = "us-east-2"
    client = boto3.client(
        service_name='secretsmanager',
        aws_access_key_id='',
        aws_secret_access_key='',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return None
    if 'SecretString' in get_secret_value_response:
        return json.loads(get_secret_value_response['SecretString'])
    else:
        decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
        return json.loads(decoded_binary_secret)

secret = get_secret()
app = FastAPI()

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client["chatHistory"]
chat_collection = db["chat"]

# AI Model
model_1 = AzureChatOpenAI(
    azure_endpoint=secret['azure_endpoint'],
    azure_deployment=secret['azure_deployment'],
    api_key=secret['api_key'],
    api_version="2024-02-15-preview",
    model_version="1",
    streaming=True,
)

@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

tools = [add]
model_1_with_tools = model_1.bind_tools(tools, tool_choice="auto")

SYSTEM_MESSAGE_CONTENT = (
    "Your are an AI agent responsible for answers user's question. You are part of RAG implementation. "
    "If the query matches vector store, your prompt will be include prompt and 3 blocks of information extracted from vectore store."
    )

class UserMessage(BaseModel):
    user_id: str
    message: str
    thread_id: str = None

class AIResponse(BaseModel):
    response: str

def initialize_chat_history(user_id: str):
    if not chat_collection.find_one({"user_id": user_id}):
        system_message = message_to_dict(SystemMessage(content=SYSTEM_MESSAGE_CONTENT))
        system_message["timestamp"] = datetime.now(timezone.utc)
        chat_collection.insert_one({"user_id": user_id, "messages": [system_message]})

def add_message_to_history(user_id: str, message: dict):
    message["timestamp"] = datetime.now(timezone.utc)
    chat_collection.update_one(
        {"user_id": user_id},
        {"$push": {"messages": message}}
    )

def get_chat_history(user_id: str) -> List[dict]:
    user_chat = chat_collection.find_one({"user_id": user_id})
    if not user_chat:
        return []
    else:
        for message in user_chat['messages']:
            if message['type'] == 'ai' and len(message['data'].get('tool_calls', [])) != 0:
                for tool_call in message['data']['tool_calls']:
                    selected_tool = {"add": add}[tool_call["name"].lower()]
                    tool_output = selected_tool.run(tool_call["args"])
    return messages_from_dict(user_chat["messages"])

def clear_chat_history(user_id: str):
    chat_collection.delete_one({"user_id": user_id})

@app.post("/chat_rag/", response_model=AIResponse)
async def chat(user_message: UserMessage):
    try:
        user_id = user_message.user_id
        usr_message = user_message.message

        initialize_chat_history(user_id)

        # Retrieve from vector DB and append context
        vector_db_search = retrieve_db(query=usr_message, threshold=0.7)
        # message = f"User Prompt: {usr_message} Vector Database Match: no match. Give your own answer "  #'\n\n' Vector Database Match {vector_db_search}"
        message = f"User Prompt: {usr_message} '\n\n' Vector Database Match {vector_db_search}"

        # Add user message to history
        human_message = message_to_dict(HumanMessage(content=message))
        add_message_to_history(user_id, human_message)

        # Get previous chat
        messages = get_chat_history(user_id)
        ai_msg = model_1_with_tools.invoke(messages)
        ai_message = message_to_dict(ai_msg)
        add_message_to_history(user_id, ai_message)

        if len(ai_msg.tool_calls) != 0:
            for tool_call in ai_msg.tool_calls:
                selected_tool = {"add": add}[tool_call["name"].lower()]
                tool_output = selected_tool.run(tool_call["args"])
                tool_message = message_to_dict(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                add_message_to_history(user_id, tool_message)

            final_messages = get_chat_history(user_id)
            full_response = model_1_with_tools.invoke(final_messages)
            add_message_to_history(user_id, message_to_dict(full_response))
            with open("response_log.txt", "a") as log_file:
                log_file.write(f"{datetime.now(timezone.utc)} - User ID: {user_id}\n")
                log_file.write(f"Response: {full_response.content}\n\n")
            return AIResponse(response=full_response.content)

        return AIResponse(response=ai_msg.content)

    except Exception as e:
        # Log traceback to console or a log file
        print("Exception in /chat/:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )
    
@app.post("/chat_direct/", response_model=AIResponse)
async def chat_direct(user_message: UserMessage):
    try:
        user_id = user_message.user_id
        usr_message = user_message.message

        initialize_chat_history(user_id)

        human_message = message_to_dict(HumanMessage(content=usr_message))
        add_message_to_history(user_id, human_message)

        messages = get_chat_history(user_id)
        ai_msg = model_1_with_tools.invoke(messages)
        ai_message = message_to_dict(ai_msg)
        add_message_to_history(user_id, ai_message)

        if len(ai_msg.tool_calls) != 0:
            for tool_call in ai_msg.tool_calls:
                selected_tool = {"add": add}[tool_call["name"].lower()]
                tool_output = selected_tool.run(tool_call["args"])
                tool_message = message_to_dict(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                add_message_to_history(user_id, tool_message)

            final_messages = get_chat_history(user_id)
            full_response = model_1_with_tools.invoke(final_messages)
            add_message_to_history(user_id, message_to_dict(full_response))
            return AIResponse(response=full_response.content)

        return AIResponse(response=ai_msg.content)

    except Exception as e:
        print("Exception in /chat_direct/:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )


@app.get("/chat_history/{user_id}")
async def get_user_chat_history(user_id: str):
    """
    Retrieve the chat history for a specific user.
    """
    messages = get_chat_history(user_id)
    if not messages:
        raise HTTPException(status_code=404, detail="No chat history found for this user.")
    return {"chat_history": messages}

@app.post("/clear_history/{user_id}")
async def clear_user_chat_history(user_id: str):
    """
    Clear the chat history for a specific user.
    """
    clear_chat_history(user_id)
    return {"message": f"Chat history for user {user_id} cleared successfully"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, log_level="info")