# This script is run during build-time to cache the model in the Docker image.
import openai_whisper as whisper


def download_model():
    global model
    model = whisper.load_model("base")


if __name__ == "__main__":
    download_model()
