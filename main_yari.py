import traceback

from openai import OpenAI
import streamlit as st

import file_parser
from configs import OAI_MODEL
from rag import RagInstance
from collections import defaultdict

st.title(f"AIce Tutor")

rag_instance = RagInstance()

pages = []

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'csv'])
if uploaded_file is not None:
    # Check if the uploaded file is a PDF
    if uploaded_file.type == "application/pdf":
        try:
            extracted_text = file_parser.from_file_to_document(uploaded_file)

            rag_instance.expand_knowledge_vector_db(extracted_text)

            st.text_area(f"Extracted Text, type {type(pages)}", extracted_text, height=300)
        except Exception as e:
            st.error("Error processing the PDF file.")
            st.write(f"error is {str(e)}")
            traceback.print_exc()
    else:
        st.write("File uploaded successfully!")

# Create a button
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OAI_MODEL

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

mode = st.sidebar.radio(
    "Choose your AI Tutor Mode:",
    ('Question', 'Answer')
)

### Logic for when we listen for user's answer
if mode == "Answer":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Try to answer:"):
        user_response = "some user response"
        st.session_state['messages'].append({"role": "user", "content": user_response})
        with st.chat_message("user"):
            st.markdown(user_response)
        model_response, messages = rag_instance.query(prompt, "intermediate", st.session_state['messages'],
                                                      is_correctness=True,
                                                      original_question=st.session_state['questions'][-1])

        st.session_state['messages'].append({"role": "assistant", "content": model_response})
        with st.chat_message("assistant"):
            st.markdown(model_response)
        # print(st.session_state['questions'], st.session_state['messages'])
else:
    mode = "Question"
    if prompt := st.chat_input("Ask a question:"):
        # Add question to session state
        st.session_state['messages'] = []
        with st.chat_message("user"):
            st.markdown(prompt)
        # full_response, messages = "Full question to user: " + prompt, st.session_state['messages'] +[{"role": "user", "content": "full_responseee : " + prompt}]

        full_response, messages = rag_instance.query(prompt, "intermediate", [])

        with st.chat_message("assistant"):
            st.markdown(full_response)

        st.session_state['questions'].append(full_response)
        st.session_state['messages'].append({"role": "assistant", "content": full_response})
        print(st.session_state['questions'], st.session_state['messages'])
        # print(st.session_state['questions'], st.session_state['messages'])
