# This script is run during build-time to cache the model in the Docker image.
from faster_whisper import WhisperModel


def download_model():
    global model
    model = WhisperModel("base")


if __name__ == "__main__":
    download_model()
