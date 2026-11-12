
import numpy as np

from pathlib import Path

EMBEDDING_DIR = Path("app/uploads/embeddings")

for image_folder in sorted(EMBEDDING_DIR.iterdir()):

    if not image_folder.is_dir():
        continue

    print(f"\nImage ID: {image_folder.name}")

    embedding_files = sorted(
        image_folder.glob("person_*.npy")
    )

    if not embedding_files:
        print("  No embeddings found")
        continue

    for file in embedding_files:

        embedding = np.load(
            file,
            allow_pickle=False
        )

        print(
            f"  {file.name} | "
            f"Shape: {embedding.shape} | "
            f"Data type: {embedding.dtype}"
        )
