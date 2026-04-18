import streamlit as st
from agent import app

st.set_page_config(page_title="Study Buddy AI")

st.title("🤖 Study Buddy AI")

# Initialize session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Input
user_input = st.chat_input("Ask your question...")

# Display previous messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)

# When user enters question
if user_input:
    # Show user message
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Get response
    res = app.invoke(
        {"question": user_input},
        config={"configurable": {"thread_id": st.session_state.thread_id}}
    )

    answer = res["answer"]

    # Show assistant response
    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)