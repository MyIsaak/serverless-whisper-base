import os
import openai_whisper as whisper
import util


# Init is run on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = whisper.load_model("base")


# Inference is run for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    audio_format, kwargs, url, error = util.parse_input_args(model_inputs)
    if error:
        return {"error": error}

    tmp_file = util.download_file(audio_format, url)

    # Run the model
    segments, info = model.transcribe(tmp_file, **kwargs)

    os.remove(tmp_file)

    return {
        "text": util.into_vtt(segments),
        "info": info
    }
