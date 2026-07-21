import cv2, json, base64
import numpy as np


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR image array."""
    raw = np.frombuffer(raw_bytes, np.uint8)
    # Note in the case of colour images, the decoded images will have the channels stored in B G R order.
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image array as PNG bytes for display."""
    return cv2.imencode(".png", image)[1].tobytes()

def format_metadata_for_llm(metadata: dict):
    """Wrapped to pass image metadata to LLM"""
    if not metadata:
        return ""
    json_str = json.dumps(metadata, default=str, indent=2)
    return (
        "The following metadata was extracted from the image file the user shared. "
        "It may not be directly relevant to their question:\n"
        f"{json_str}\n\n"
    )

def build_image_message(text, file_upload, metadata=None):
    """
    Encode an image upload in a OpenAI spec message dict alongside user text prompt
    Include metadata if supplied
    """
    full_text = format_metadata_for_llm(metadata) + text

    if not file_upload.value:
        return {"role": "user", "content": full_text}

    image_data = base64.b64encode(
        file_upload.value[0].contents
    ).decode('utf-8')

    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            },
            {
                "type": "text",
                "text": full_text
            }
        ]
    }
