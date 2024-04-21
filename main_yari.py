import traceback

from openai import OpenAI
import streamlit as st

import file_parser
from configs import OAI_MODEL
from rag import RagInstance
from collections import defaultdict

st.title(f"AIce Tutor")

pages = []

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'csv'])
if uploaded_file is not None:
    # Check if the uploaded file is a PDF
    if uploaded_file.type == "application/pdf":
        try:
            extracted_text = file_parser.from_file_to_document(uploaded_file)

            # rag_instance.expand_knowledge_vector_db(extracted_text)

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
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        rag_instance = RagInstance()
        retrieved_docs = file_parser.get_all_processed_files()
        rag_instance.expand_knowledge_vector_db(retrieved_docs)

        model_response, messages, used_docs = rag_instance.query(prompt, "intermediate", st.session_state['messages'],
                                                                 is_correctness=True,
                                                                 original_question=st.session_state['questions'][-1])

        st.session_state['messages'].append({"role": "assistant", "content": model_response})

        with st.chat_message("assistant"):
            st.markdown(model_response)
            for source in (f"[{i + 1}] {doc}" for i, doc in enumerate(used_docs)):
                st.markdown(source)

        # print(st.session_state['questions'], st.session_state['messages'])
else:
    mode = "Question"
    if prompt := st.chat_input("Ask a question:"):
        # Add question to session state
        st.session_state['messages'] = []
        with st.chat_message("user"):
            st.markdown(prompt)
        # full_response, messages = "Full question to user: " + prompt, st.session_state['messages'] +[{"role": "user", "content": "full_responseee : " + prompt}]

        rag_instance = RagInstance()
        retrieved_docs = file_parser.get_all_processed_files()
        rag_instance.expand_knowledge_vector_db(retrieved_docs)

        model_response, messages, used_docs = rag_instance.query(prompt, "intermediate", [])

        with st.chat_message("assistant"):
            st.markdown(model_response)
            for source in (f"[{i + 1}] {doc}" for i, doc in enumerate(used_docs)):
                st.markdown(source)

        st.session_state['questions'].append(model_response)
        st.session_state['messages'].append({"role": "assistant", "content": model_response})
        print(st.session_state['questions'], st.session_state['messages'])
        # print(st.session_state['questions'], st.session_state['messages'])
