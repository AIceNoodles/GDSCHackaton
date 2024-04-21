# SocrAItes

SocrAItes is a RAG-powered application that guides users of all ages and backgrounds through an active process of discovery and learning, by asking targeted questions about a source and providing constructive feedback. Grounded in cognitive science research, our maieutic approach to education allows the user to build and test their understanding, accompanied by a knowledgeable tutor, while never leaving the driver seat.

## App Workflow

Within our app, users can set and keep track of specific learning objectives through a dedicated page. They can interact with the system by typing into a chat, or directly speaking into a microphone. The learning begins by uploading material that they wish to learn about in the form of a text, audio or video file. Then, the user specifies some aspects of the material that they would like to understand better, and tune the desired difficulty level. Our system provides targeted questions and exercises, and then guides the user towards a solution.

## Under the hood

To craft meaningful questions and ensure factuality of the feedback, SocrAItes leverages a cutting-edge Retrieval Augmented Generation system, which has been shown to drastically reduce hallucinations and lead to well-funded answers. We build a vector database of semantic embeddings based on the provided material and use retrieval to help the LLM guide the user, using a series of carefully crafted prompts and resorting to the OpenAI API to achieve state-of-the-art performance.

## Demo

To install our project, navigate to the root directory and run:

```
pip install -r requirements.txt
```

We have built a fully working proof of concept using Streamlit. To start it locally, run:

```
streamlit run main.py
```
