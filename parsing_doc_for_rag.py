import os
import subprocess
from langchain.docstore.document import Document as doc


#### Running Nougat to get the mmd in a folder called nougat_output 
output_dir = "nougat_output"
input_dir = "uploaded_chapters"
os.makedirs(output_dir, exist_ok=True)
command = ["nougat", input_dir, "-o", output_dir]

result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("Nougat processed successfully.")
    print("Output:", result.stdout)
else:
    print("Error running Nougat:")
    print(result.stderr)
#### 

#### Helper Function for creating the input for RAG
def read_mmd_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".mmd"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
    return texts
#### 

### Creating the drocument
mmd_texts = read_mmd_files('nougat_output')
