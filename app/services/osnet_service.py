
import numpy as np
import torch

from pathlib import Path
from pyppbox_torchreid.utils import FeatureExtractor

from app.services.image_quality_service import ImageQualityService


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
PERSON_DIR = BASE_DIR / "uploads" / "persons"
EMBEDDING_DIR = BASE_DIR / "uploads" / "embeddings"


class OSNetFeatureService:

    _extractor = None

    @classmethod
    def get_extractor(cls):

        if cls._extractor is None:

            model_files = list(
                MODEL_DIR.glob("osnet_x1_0*.pth")
            )

            if not model_files:
                raise FileNotFoundError(
                    "Pretrained OSNet model not found in app/models"
                )

            model_path = model_files[0]

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            print(f"Loading OSNet model on {device}...")
            print(f"Model path: {model_path}")

            cls._extractor = FeatureExtractor(
                base_model_name="osnet_x1_0",
                model_path=str(model_path),
                device=device,
                verbose=True
            )

            print("OSNet model loaded successfully!")

        return cls._extractor

    @classmethod
    def save_embeddings(cls, image_id, embeddings):

        image_embedding_folder = EMBEDDING_DIR / image_id

        image_embedding_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove old embeddings to avoid stale results
        for old_file in image_embedding_folder.glob("person_*.npy"):
            old_file.unlink()

        for item in embeddings:

            person_number = item["person_number"]
            embedding = item["embedding"]

            embedding_path = (
                image_embedding_folder
                / f"person_{person_number}.npy"
            )

            np.save(embedding_path, embedding)

        print(
            f"Saved {len(embeddings)} embeddings "
            f"to {image_embedding_folder}"
        )

        return str(image_embedding_folder)

    @classmethod
    def extract_features(cls, image_id: str):

        person_folder = PERSON_DIR / image_id

        if not person_folder.exists():
            raise ValueError(
                "Person crops not found. Run YOLO detection first."
            )

        person_images = sorted(
            person_folder.glob("person_*.jpg"),
            key=lambda path: int(path.stem.split("_")[1])
        )

        if not person_images:
            raise ValueError(
                "No person images found for feature extraction"
            )

        # 1. Validate every person crop
        valid_persons = []
        skipped_persons = []

        for image_path in person_images:

            person_number = int(
                image_path.stem.split("_")[1]
            )

            quality_result = (
                ImageQualityService.validate_crop(image_path)
            )

            person_result = {
                "person_number": person_number,
                "quality": quality_result
            }

            if quality_result["can_process"]:

                valid_persons.append({
                    "person_number": person_number,
                    "image_path": image_path,
                    "quality": quality_result
                })

            else:

                skipped_persons.append(person_result)

        # 2. Handle case where all crops fail validation
        if not valid_persons:

            embedding_folder = cls.save_embeddings(
                image_id,
                []
            )

            return {
                "image_id": image_id,
                "total_detected_persons": len(person_images),
                "total_processed_persons": 0,
                "total_skipped_persons": len(skipped_persons),
                "embedding_dimension": None,
                "embedding_folder": embedding_folder,
                "persons": [],
                "skipped_persons": skipped_persons
            }

        # 3. Extract features only from valid crops
        extractor = cls.get_extractor()

        image_paths = [
            str(person["image_path"])
            for person in valid_persons
        ]

        features = extractor(image_paths)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        features = np.asarray(features)

        if (
            features.ndim != 2
            or features.shape[0] != len(valid_persons)
        ):
            raise ValueError(
                "Unexpected OSNet feature output shape"
            )

        embeddings = []
        saved_embeddings = []

        # 4. Normalize and prepare embeddings
        for index, feature in enumerate(features):

            norm = np.linalg.norm(feature)

            if not np.isfinite(norm) or norm == 0:
                raise ValueError(
                    "OSNet generated an invalid embedding"
                )

            normalized_feature = feature / norm

            person = valid_persons[index]

            person_number = person["person_number"]

            saved_embeddings.append({
                "person_number": person_number,
                "embedding": normalized_feature.astype(
                    np.float32
                )
            })

            embeddings.append({
                "person_number": person_number,
                "embedding_dimension": int(
                    len(normalized_feature)
                ),
                "embedding": np.round(
                    normalized_feature, 6
                ).tolist(),
                "quality": person["quality"]
            })

        # 5. Save embeddings using original person numbers
        embedding_folder = cls.save_embeddings(
            image_id,
            saved_embeddings
        )

        # 6. Return final response
        return {
            "image_id": image_id,
            "total_detected_persons": len(person_images),
            "total_processed_persons": len(embeddings),
            "total_skipped_persons": len(skipped_persons),
            "embedding_dimension": int(features.shape[1]),
            "embedding_folder": embedding_folder,
            "persons": embeddings,
            "skipped_persons": skipped_persons
        }
