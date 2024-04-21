import os

#### Running Nougat to get the mmd in a folder called nougat_output
output_dir = "nougat_output"
input_dir = "uploaded_chapters"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)


def run_nougat():
    os.makedirs(output_dir, exist_ok=True)

    result = None

    for filename in os.listdir(input_dir):
        if "pdf" not in filename:
            continue
        command = ["nougat", input_dir + "/" + filename, "-o", output_dir]
        output_file = output_dir + "/" + filename.split(".")[-2] + ".mmd"

        print(command)
        print(output_file)
        if os.path.exists(output_file):
            print("Skipping", filename, "as it has already been processed.")
            continue
        else:
            print("Processing", filename)
            import subprocess
            result = subprocess.run(command, capture_output=True, text=True)

    if result is None or result.returncode == 0:
        print("Nougat processed successfully.")
        print("Output:", result is None or result.stdout)
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
