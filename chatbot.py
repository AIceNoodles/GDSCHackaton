from dataclasses import dataclass

import streamlit as st
from vertexai.preview.language_models import TextGenerationModel


@dataclass
class Message:
    actor: str
    payload: str


def get_llm() -> TextGenerationModel:
    return TextGenerationModel.from_pretrained("text-bison@001")


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi!How can I help you?")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    response: str = get_llm().predict(prompt=prompt).text
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)