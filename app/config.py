import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = os.getenv("APP_NAME", "Person Re-Identification API")
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 1024))
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 25000000))


settings = Settings()