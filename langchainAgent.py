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

app = FastAPI()

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client["chatHistory"]
chat_collection = db["chat"]

# Initialize the AI Model
"""model_1 = AzureChatOpenAI(
    azure_endpoint='',
    azure_deployment="",
    api_key='',
    api_version="",
    model_version="",
    streaming=,
)"""

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
    "Your are an AI agent responsible for answers user's question regarding legal statues."
)

# Define request and response models
class UserMessage(BaseModel):
    user_id: str
    message: str

class AIResponse(BaseModel):
    response: str

# Helper Functions for MongoDB
def initialize_chat_history(user_id: str):
    """Initialize chat history for a user if it doesn't exist."""
    if not chat_collection.find_one({"user_id": user_id}):
        system_message = message_to_dict(SystemMessage(content=SYSTEM_MESSAGE_CONTENT))#.to_dict()
        timestamp = datetime.now(timezone.utc)
        system_message["timestamp"] = timestamp
        chat_collection.insert_one({"user_id": user_id, "messages": [system_message]})

def add_message_to_history(user_id: str, message: dict):
    """Add a message to a user's chat history."""
    timestamp = datetime.now(timezone.utc)
    message["timestamp"] = timestamp
    
    chat_collection.update_one(
        {"user_id": user_id},
        {"$push": {"messages": message}}
    )

def get_chat_history(user_id: str) -> List[dict]:
    """Retrieve the chat history for a user."""
    user_chat = chat_collection.find_one({"user_id": user_id})
    if not user_chat:
        return []
    else:
        for message in user_chat['messages']:
            if message['type'] == 'ai' and len(message['data']['tool_calls'])!=0:
                for tool_call in message['data']['tool_calls']:
                    selected_tool = {"add":add}[tool_call["name"].lower()]
                    tool_output = selected_tool.run(tool_call["args"]['__arg1'])

    return messages_from_dict(user_chat["messages"])

def clear_chat_history(user_id: str):
    """Clear the chat history for a user."""
    chat_collection.delete_one({"user_id": user_id})

@app.post("/chat/", response_model=AIResponse)
async def chat(user_message: UserMessage):
    user_id = user_message.user_id
    message = user_message.message

    # Initialize user chat history if not exists
    initialize_chat_history(user_id)

    # Add user message to chat history
    human_message = message_to_dict(HumanMessage(content=message))#.to_dict()
    #human_message = {"type":human_message.type, "content":human_message.content}
    add_message_to_history(user_id, human_message)

    # Retrieve chat history and reconstruct messages
    messages = get_chat_history(user_id)

    # Process user message
    ai_msg = model_1_with_tools.invoke(messages)

    #if ai_response_content:
    ai_message = message_to_dict(ai_msg)#.to_dict()
    add_message_to_history(user_id, ai_message)

    if len(ai_msg.tool_calls)!=0:
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"add":add}[tool_call["name"].lower()]
            tool_output = selected_tool.run(tool_call["args"]['__arg1'])
            tool_message = message_to_dict(ToolMessage(tool_output, tool_call_id=tool_call["id"]))#.to_dict()
            #tool_message = {"type":tool_message.type, "content":tool_message.content}
            toolargs = tool_call["args"]['__arg1']
            toolMessageTest = f'print({toolargs})'
            add_message_to_history(user_id, tool_message)

        # Generate final response
        final_messages = get_chat_history(user_id)
        full_response = model_1_with_tools.invoke(final_messages)
        ai_response_content = full_response.content
        ai_message = message_to_dict(full_response)#.to_dict()
        #ai_message = {"type":ai_message.type, "content":ai_message.content}
        add_message_to_history(user_id, ai_message)
        return AIResponse(response=ai_response_content)
    return AIResponse(response=ai_msg.content)

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