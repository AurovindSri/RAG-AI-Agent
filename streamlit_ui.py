import streamlit as st
import requests
import json

# API Endpoint
CHAT_API_URL = "http://localhost:8080/chat/"
CHAT_HISTORY_URL = "http://localhost:8080/chat_history/"
CLEAR_HISTORY_URL = "http://localhost:8080/clear_history/"

# Helper functions
def send_message(user_id, message):
    """Send a message to the chatbot."""
    response = requests.post(CHAT_API_URL, json={"user_id": user_id, "message": message})
    if response.status_code == 200:
        try:
            return response.json()["response"]
        except (ValueError, KeyError):
            st.error("Invalid response format from the server.")
            return None
    else:
        try:
            error_detail = response.json().get('detail', 'Failed to send message')
        except ValueError:
            error_detail = f"Non-JSON response received. Status code: {response.status_code}"
        st.error(f"Error: {error_detail}")
        return None

def get_chat_history(user_id):
    """Fetch chat history for a user."""
    response = requests.get(f"{CHAT_HISTORY_URL}{user_id}")
    if response.status_code == 200:
        return response.json()["chat_history"]
    else:
        st.warning("No chat history found.")
        return []

def clear_chat_history(user_id):
    """Clear the chat history for a user."""
    response = requests.post(f"{CLEAR_HISTORY_URL}{user_id}")
    if response.status_code == 200:
        st.success("Chat history cleared successfully.")
    else:
        st.error("Failed to clear chat history.")

# Streamlit UI
st.title("Approval Engine AI Agent")
st.sidebar.title("Settings")

# User ID input
user_id = st.sidebar.text_input("User ID", value="user1")

# Reset session state and chat history when the user ID changes
if "user_id" not in st.session_state or st.session_state["user_id"] != user_id:
    st.session_state["user_id"] = user_id
    st.session_state["messages"] = get_chat_history(user_id)

# Display chat history (Excluding system and tool messages)
st.subheader("Chat History")
for msg in st.session_state["messages"]:
    if msg["type"] == "human":
        # Apply blue color to "You" label and increase font size for label and content
        st.markdown(f"<p><strong style='color: blue; font-size: 18px;'>You:</strong> <span style='font-size: 16px;'>{msg['content']}</span></p>", unsafe_allow_html=True)
    elif msg["type"] == "ai" and msg['content'].strip():
        # Only display AI response if it is not empty and apply green color and increase font size for label and content
        st.markdown(f"<p><strong style='color: green; font-size: 18px;'>Bot:</strong> <span style='font-size: 16px;'>{msg['content']}</span></p>", unsafe_allow_html=True)

# Form for user message input with automatic send on Enter key
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:")
    submit_button = st.form_submit_button(label="Send")

    # Avoid duplicate addition of messages
    if submit_button and user_input.strip():
        # Send message to API
        ai_response = send_message(user_id, user_input)
        if ai_response:
            # Update local session messages only if new message is sent and AI response is received
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"].append({"type": "human", "content": user_input})
            st.session_state["messages"].append({"type": "ai", "content": ai_response})
            st.rerun()

# Clear chat history button
if st.sidebar.button("Clear Chat History"):
    clear_chat_history(user_id)
    st.session_state["messages"] = []
    st.rerun()