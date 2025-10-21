import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "Main/dataset/train.csv"
OUTPUT_CSV = "stella_embeddings.csv"
MODEL_NAME = "NovaSearch/stella_en_1.5B_v5"
BATCH_SIZE = 256 # Adjust based on GPU VRAM and text length

def extract_text_content(catalog_content):
    """
    Extracts and cleans Item Name and Bullet Points from the catalog content string.
    """
    if not isinstance(catalog_content, str):
        return ""
        
    lines = catalog_content.strip().split('\n')
    text_parts = []
    
    for line in lines:
        if line.startswith("Item Name:"):
            text_parts.append(line.replace("Item Name:", "").strip())
        elif line.startswith("Bullet Point"):
            bullet_text = re.sub(r'^Bullet Point \d+:\s*', '', line).strip()
            text_parts.append(bullet_text)
            
    # Join parts with a period and space for clear separation.
    text = ". ".join(filter(None, text_parts))
    # Final cleanup for any resulting artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Execution ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # 2. Preprocess all texts in a vectorized operation
    print("Preprocessing texts...")
    # Use tqdm with pandas .apply for progress tracking
    tqdm.pandas(desc="Cleaning text")
    texts_to_encode = df['catalog_content'].progress_apply(extract_text_content).tolist()
    ids = df['sample_id'].tolist()
    print("Text preprocessing complete.")

    # 3. Load model with optimizations
    print(f"\nLoading model: {MODEL_NAME}")
    # Load directly to GPU and use float16 for A100 performance
    model = SentenceTransformer(
        MODEL_NAME, 
        trust_remote_code=True, 
        device=device,
        torch_dtype=torch.float16 
    )
    print("Model loaded successfully to GPU in fp16 mode.")

    # 4. Encode texts in a single batch operation
    print(f"\nEncoding {len(texts_to_encode)} texts with batch size {BATCH_SIZE}...")
    embeddings = model.encode(
        texts_to_encode,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("Encoding complete.")

    # 5. Save results
    print(f"\nCreating DataFrame with {len(embeddings)} embeddings...")
    emb_df = pd.DataFrame(embeddings)
    emb_df.insert(0, "sample_id", ids)
    emb_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Saved embeddings to {OUTPUT_CSV}")
    print(f"Output shape: {emb_df.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")