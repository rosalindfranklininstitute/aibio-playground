import cv2, json, base64
import numpy as np

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into an RGB image array (currently unused by project)."""
    raw = np.frombuffer(raw_bytes, np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def encode_png(image: np.ndarray) -> bytes:
    """Encode an image array as PNG bytes. Assumes RGB channel order for 3-channel images."""
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

def resize_for_display(arr: np.ndarray, min_dim: int = 400, max_dim: int = 1000) -> np.ndarray:
    """Resize a (Y,X,C) uint8 array so its longest side sits within
    [min_dim, max_dim], scaling up if smaller than min_dim, down if larger
    than max_dim, and leaving it untouched if already within range."""
    from skimage.transform import resize
    from skimage.util import img_as_float, img_as_ubyte, img_as_uint

    type = arr.dtype

    y, x = arr.shape[:2]
    longest = max(y, x)
    if min_dim <= longest <= max_dim:
        return arr
    target = min_dim if longest < min_dim else max_dim
    scale = target / longest
    new_shape = (max(1, int(y * scale)), max(1, int(x * scale)))
    out_shape = new_shape if arr.ndim == 2 else (*new_shape, arr.shape[2])

    if np.issubdtype(type, np.integer) == True:
        arr = img_as_float(arr)
        
    resized = resize(arr, out_shape, anti_aliasing=True, preserve_range=True)

    if type == np.uint8:
        out = img_as_ubyte(resized)
    elif type == np.uint16:
        out = img_as_uint(resized)
    else:
        out = resized
    return out
