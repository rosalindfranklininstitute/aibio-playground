import cv2
import numpy as np


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR image array."""
    raw = np.frombuffer(raw_bytes, np.uint8)
    # Note in the case of colour images, the decoded images will have the channels stored in B G R order.
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image array as PNG bytes for display."""
    return cv2.imencode(".png", image)[1].tobytes()
