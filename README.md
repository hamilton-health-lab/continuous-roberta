# Continuous-RoBERTa  😄  😨  😡  😞
Sentiment analysis tools for estimating the intensity of four emotions from text. 

Validated on self-reported affect from dream reports. See article from [SLEEP]((https://academic.oup.com/sleep/article-abstract/47/12/zsae210/7754282))

This repository contains a Python script to analyze dream reports (or any text of linguistic context) for emotional intensities using pre-trained transformers models (cardiffnlp/twitter-roberta-base-2022-154m). The script processes the input text and computes the intensity of emotions (relative values between 0-1): joy, fear, anger, and sadness.

Fine-tuning data come from Mohammad et al., (2018) [SemEval-2018 Task 1: Affect in Tweets](https://aclanthology.org/S18-1001/). We thank the authors for making these data openly available.

Validation metrics and the UI spaces for testing are available on huggingface 🤗 here: [joy](https://huggingface.co/garrettbaber/twitter-roberta-base-joy-intensity), [fear](https://huggingface.co/garrettbaber/twitter-roberta-base-fear-intensity), [anger](https://huggingface.co/garrettbaber/twitter-roberta-base-anger-intensity), [sadness](https://huggingface.co/garrettbaber/twitter-roberta-base-sadness-intensity)

## Getting Started

### Formatting your CSV file

Prepare your CSV by naming the column containing text (e.g., dream reports) as "text" in all lowercase. 

### Prerequisites

Ensure you have the following packages installed:
- transformers
- pandas
- torch

You can install these using pip:

```bash
pip install transformers pandas torch
```

### Body of the Code

#### Ensure you modify the read-in file and write/save paths

```bash
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

# Load the dataset in chunks
url = "insert_read_path.csv" # CHANGE TO YOUR FILE PATH
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

# Save the new dataframe to the specified output path
output_file_path = "insert_save_path.csv" # CHANGE TO YOUR FILE PATH
final_results.to_csv(output_file_path, index=False)
```
