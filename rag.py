from typing import List, Optional
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from openai import OpenAI

client = OpenAI()
from prompt_generator import assemble_prompt, assemble_correctness_prompt

# files: list[dict]  # has a key "text" and other metadata keys

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def get_response_message(completion) -> str:
    return completion.choices[0].message.content


class RagInstance:
    def __init__(self):
        self.knowledge_vect_db = None
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity

        )

    def create_vec_db(self, files):
        RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=file["text"], metadata={k: v for k, v in file.items() if k != "text"})
            for file in files
        ]

        def split_documents(
                chunk_size: int,
                knowledge_base: List[LangchainDocument],
                tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
        ) -> List[LangchainDocument]:
            """
            Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
            """
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(tokenizer_name),
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10),
                add_start_index=True,
                strip_whitespace=True,
                separators=MARKDOWN_SEPARATORS,
            )

            docs_processed = []
            for doc in knowledge_base:
                docs_processed += text_splitter.split_documents([doc])

            # Remove duplicates
            unique_texts = {}
            docs_processed_unique = []
            for doc in docs_processed:
                if doc.page_content not in unique_texts:
                    unique_texts[doc.page_content] = True
                    docs_processed_unique.append(doc)

            return docs_processed_unique

        docs_processed = split_documents(
            512,  # We choose a chunk size adapted to our model
            RAW_KNOWLEDGE_BASE,
            tokenizer_name=EMBEDDING_MODEL_NAME,
        )

        knowledge_vector_database = FAISS.from_documents(
            docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

        return knowledge_vector_database

    knowledge_vect_db = None

    def expand_knowledge_vector_db(self, files):
        if self.knowledge_vect_db is None:
            self.knowledge_vect_db = self.create_vec_db(files)
        else:
            self.knowledge_vect_db.merge_from(self.create_vec_db(files))

    # user_query: str  # the user's query
    # expertise_level: str  # the user's expertise level
    # messages: list[dict]  # the conversation history
    # model: str  # the model to use for the completion

    def query(self, user_query, expertise_level, messages, is_correctness=False, original_question=None):
        if self.knowledge_vect_db:
            retrieved_docs = self.knowledge_vect_db.similarity_search(query=user_query, k=5)
        else:
            retrieved_docs = []

        if is_correctness is False:
            new_message = assemble_prompt(
                [doc.page_content for doc in retrieved_docs],
                user_query,
                expertise_level=expertise_level,
            )
        else:
            new_message = assemble_correctness_prompt(
                [doc.page_content for doc in retrieved_docs],
                model_question=original_question,
                user_input=user_query,
                expertise_level=expertise_level,
            )

        messages = messages + [{"role": "user", "content": new_message}]

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response = get_response_message(completion)

        # response = "some response (test purposes)"

        messages = messages + [{"role": "assistant", "content": response}]

        return response, messages, [doc.page_content for doc in retrieved_docs]
