from typing import List
import json

rag_instructions = (
    "Given the context information and not prior knowledge, answer the query asking about citations over different topics.\n"
    "Please provide your answer in the form of a structured JSON format containing a list of authors as the citations. Some examples are given below."
    "\n\nQuery: Which citation discusses the impact of safety RLHF measured by reward model score distributions?\nResponse: {'citations': [{'author': 'Llama 2: Open Foundation and Fine-Tuned Chat Models', 'year': 24, 'desc': 'Impact of safety RLHF measured by reward model score distributions. Left: safety reward model scores of generations on the Meta Safety test set. The clustering of samples in the top left corner suggests the improvements of model safety. Right: helpfulness reward model scores of generations on the Meta Helpfulness test set.'}]}\n\nQuery: Which citations are mentioned in the section on RLHF Results?\nResponse: {'citations': [{'author': 'Gilardi et al.', 'year': 2023, 'desc': ''}, {'author': 'Huang et al.', 'year': 2023, 'desc': ''}]}\n")


def prompt_generator(rag_output: List[str], user_input: str, rag_instructions: str = rag_instructions):
    return (rag_instructions +
            "User Input: \n" + user_input +
            "\n\nRAG Response: \n" + "\n".join(rag_output))

