import requests
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = whisper.load_model("base", device="cuda", in_memory=True, fp16=True)

def _parse_arg(args : str, data : dict, default = None, required = False):
    arg = data.get(args, None)
    if arg == None:
        if required:
            raise Exception(f"Missing required argument: {args}")
        else:
            return default

    return arg

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    try:
        url = _parse_arg("url", model_inputs, required=True)
        audio_format = _parse_arg("format", model_inputs, "mp3")
        kwargs = _parse_arg("kwargs", model_inputs, {})

    except Exception as e:
        return {"error":str(e)}

    audio_buffer = BytesIO(requests.get(url).content)

    tmp_file = "input."+audio_format
    with open(tmp_file, 'wb') as file:
        file.write(audio_buffer.getbuffer())
    
    # Run the model
    result = model.transcribe(tmp_file, fp16=True, **kwargs)
    result['segments'] = [{
        "id":x['id'],
        "seek":x['seek'],
        "start":x['start'],
        "end":x['end'],
        "text":x['text']
        } for x in result['segments']]
    os.remove(tmp_file)
    # Return the results as a dictionary
    return result
