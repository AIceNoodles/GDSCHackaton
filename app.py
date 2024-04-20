import streamlit as st
from transformers import  AutoTokenizer, AutoModelForCausalLM
import torch
import os 
from datetime import datetime
import pandas as pd 


hf_token = os.getenv('HF_TOKEN')

name = 'google/gemma-2b-it'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)


def get_response(prompt_text):
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")

    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('Response:------------------')
    print(response_text)
    return response_text

st.title('Critical Reasoning Assistant using Gemma')
user_input = st.text_area("Enter text here to analyze and generate a response:", height=150)
if st.button('Generate Response'):
    if user_input:
        with st.spinner('AI is at work...'):
            response = get_response(user_input)
            st.write(response)
    else:
        st.error("Please enter some text to analyze and generate a response from.")

