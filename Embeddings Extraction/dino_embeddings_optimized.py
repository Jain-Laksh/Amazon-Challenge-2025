from huggingface_hub import login
access_token = ""
login(token=access_token)


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import numpy as np

# --- Configuration ---
INPUT_CSV = "Main/dataset/train.csv"
OUTPUT_CSV = "dino_embeddings.csv"
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BATCH_SIZE = 128  # ViT-L is large; start with 128 and adjust based on VRAM
NUM_WORKERS = 8   # Adjust based on available CPU cores

# --- Dataset Definition ---
class ImageURLDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image, url
        except Exception:
            # Return None on failure, will be filtered by collate_fn
            return None, url

def collate_fn(batch):
    # Filter out samples that failed to load
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    
    images, urls = zip(*batch)
    return images, urls

# --- Main Execution ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For private models, log in via the terminal: `huggingface-cli login`
    
    # Load data URLs
    df = pd.read_csv(INPUT_CSV)
    image_links = df["image_link"].tolist()
    print(f"Loaded {len(image_links)} image links from {INPUT_CSV}")

    # Load model and processor
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device).half().eval() # Use .half() for fp16 acceleration on A100
    print("Model loaded successfully to GPU in fp16 mode")

    # Create Dataset and DataLoader
    dataset = ImageURLDataset(image_links)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True, # Helps speed up CPU to GPU data transfer
    )

    all_embeddings = []
    all_links = []

    print(f"\nProcessing {len(image_links)} images with batch size {BATCH_SIZE}...")
    for image_batch, url_batch in tqdm(dataloader):
        if image_batch is None:
            continue

        inputs = processor(images=image_batch, return_tensors="pt")
        # Move inputs to GPU and convert to float16
        inputs = {k: v.to(device).half() for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
        
        # DINOv3 uses the pooler_output, which is the [CLS] token embedding
        pooled_output = outputs.pooler_output.cpu().float().numpy()
        
        all_embeddings.append(pooled_output)
        all_links.extend(url_batch)

    # Combine and save results
    print("\nProcessing complete. Creating final DataFrame...")
    final_embeddings = np.vstack(all_embeddings)
    emb_df = pd.DataFrame(final_embeddings)
    emb_df.insert(0, "image_link", all_links)

    emb_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(emb_df)} embeddings to {OUTPUT_CSV}")
    print(f"Output shape: {emb_df.shape}")