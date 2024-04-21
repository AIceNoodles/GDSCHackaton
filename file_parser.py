import os
from typing import Dict, List

from parsing_doc_for_rag import run_nougat, output_dir
from streamlit.runtime.uploaded_file_manager import UploadedFile


def from_file_to_document(file: UploadedFile):
    run_nougat()

    output_name = file.name.split(".")[-2] + ".mmd"

    with open(f"{output_dir}/{output_name}", 'r', encoding='utf-8') as f:
        data = f.read()

    return [{
        "text": data,
        "name": output_name
    }]


def get_all_processed_files() -> List[Dict]:
    files = []

    for file in os.listdir(output_dir):
        if file.endswith(".mmd"):
            with open(f"{output_dir}/{file}", 'r', encoding='utf-8') as f:
                data = f.read()

            files.append({
                "text": data,
                "name": file
            })

    return files