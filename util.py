from io import BytesIO

import requests
from faster_whisper.transcribe import Segment


def _parse_arg(args: str, data: dict, default=None, required=False):
    arg = data.get(args, None)
    if arg is None:
        if required:
            raise ValueError(f"Missing required argument: {args}")
        else:
            return default

    return arg


def parse_input_args(model_inputs: dict):
    try:
        url = _parse_arg("url", model_inputs, required=True)
        audio_format = _parse_arg("format", model_inputs, "mp3")
        kwargs = _parse_arg("kwargs", model_inputs, {})
        return audio_format, kwargs, url, None
    except Exception as e:
        return None, None, None, str(e)


def into_vtt(segments: [Segment]):
    vtt = "WEBVTT\n\n"
    for segment in segments:
        vtt += f"{segment.start:.3f} --> {segment.end:.3f}\n"
        vtt += f"{segment.text}\n\n"

    return {"vtt": vtt}


def download_file(audio_format, url):
    audio_buffer = BytesIO(requests.get(url).content)
    tmp_file = "input." + audio_format
    with open(tmp_file, 'wb') as file:
        file.write(audio_buffer.getbuffer())
    return tmp_file
