pip install transformers pandas torch


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Define model names
model_names = [
    "garrettbaber/twitter-roberta-base-joy-intensity",
    "garrettbaber/twitter-roberta-base-fear-intensity",
    "garrettbaber/twitter-roberta-base-anger-intensity",
    "garrettbaber/twitter-roberta-base-sadness-intensity"
]

# Load tokenizers and models
tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]
models = [AutoModelForSequenceClassification.from_pretrained(name) for name in model_names]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models:
    model.to(device)
    model.eval()

# Load the dataset from the URL in chunks
url = “insert_read_path.csv"
chunk_size = 100000  # Adjust the chunk size as needed
data_reader = pd.read_csv(url, chunksize=chunk_size)

# Create an empty list to store results
results = []

with torch.no_grad():  # Disable gradient calculation
    for chunk in data_reader:
        texts = chunk["text"].tolist()

        # Initialize a dictionary to store results for this chunk
        chunk_results = {"text": texts}

        for tokenizer, model, emotion in zip(tokenizers, models, ["Joy", "Fear", "Anger", "Sadness"]):
            encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
            outputs = model(**encoded_inputs)
            intensity_values = outputs.logits.squeeze().tolist()

            # Store the results in the chunk results dictionary
            chunk_results[emotion] = intensity_values

        # Append the chunk results to the main results list
        results.append(pd.DataFrame(chunk_results))

# Concatenate all chunks into a single DataFrame
final_results = pd.concat(results, ignore_index=True)

# Save the new dataframe to Google Drive
output_file_path = ‘insert_save_path.csv’
final_results.to_csv(output_file_path, index=False)
