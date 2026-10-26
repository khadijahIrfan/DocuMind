
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "output" / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "output" / "embeddings.npy"
METADATA_FILE = BASE_DIR / "output" / "embedding_metadata.json"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # 1. Load existing text chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("No text chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError("One or more chunks contain empty text.")

    print(f"Loaded {len(texts)} text chunks.")

    # 2. Load the pretrained embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Convert text into normalized numerical vectors
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 4. Save embedding vectors
    np.save(EMBEDDINGS_FILE, embeddings)

    # 5. Save text and source-page metadata
    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "source_pages": chunk["source_pages"],
            "word_count": chunk["word_count"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6. Report results
    print("\nEmbedding generation completed!")
    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Dimensions per embedding: {embeddings.shape[1]}")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
