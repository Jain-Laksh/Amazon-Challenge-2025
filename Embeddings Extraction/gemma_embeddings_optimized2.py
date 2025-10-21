import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from huggingface_hub import login
access_token = "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"
login(token=access_token)


import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import numpy as np

# Paths
input_csv = "test.csv"
output_csv = "gemma_embeddings_test.csv"

# Load CSV
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} rows from {input_csv}")

def extract_text_content(catalog_content):
    if not isinstance(catalog_content, str):
        return ""
    lines = catalog_content.strip().split('\n')
    text_parts = []
    for line in lines:
        line = line.strip()
        if line.startswith("Item Name:"):
            text_parts.append(line.replace("Item Name:", "").strip())
        elif line.startswith("Bullet Point"):
            bullet_text = re.sub(r'^Bullet Point \d+:\s*', '', line).strip()
            text_parts.append(bullet_text)
        elif line.startswith("Value:") or line.startswith("Unit:"):
            continue
    cleaned_parts = []
    for part in text_parts:
        part = part.strip()
        if part and part[-1] not in '.!?':
            part += '.'
        cleaned_parts.append(part)
    text = " ".join(cleaned_parts)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)
    return text

# Hugging Face login
from huggingface_hub import login
access_token = input("Enter your Hugging Face access token: ")
login(token=access_token)

# Load model
model_name = "google/embeddinggemma-300m"
print(f"\nLoading model: {model_name}")
model = SentenceTransformer(model_name)
print("Model loaded successfully")

# Preprocess texts
print("Preprocessing texts...")
tqdm.pandas(desc="Cleaning text")
texts = df['catalog_content'].progress_apply(extract_text_content).tolist()
ids = df['sample_id'].tolist()
print("Text preprocessing complete.")

# Encode in batches
BATCH_SIZE = 512
embeddings = []

print(f"\nEncoding {len(texts)} texts in batches of {BATCH_SIZE}...")
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batch encoding"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode_document(batch_texts)  # Gemma-specific
    embeddings.append(batch_embeddings)

# Concatenate all batches
embeddings = np.vstack(embeddings)

# Save to CSV
emb_df = pd.DataFrame(embeddings)
emb_df.insert(0, "sample_id", ids)
emb_df.to_csv(output_csv, index=False)

print(f"Saved embeddings to {output_csv}")
print(f"Output shape: {emb_df.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
