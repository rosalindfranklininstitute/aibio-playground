METADATA = {
    "name": "segment_threshold",
    "description": (
        "Segment an image into foreground and background using pixel intensity. "
        "Use this when the user wants to isolate objects from the background "
        "based on brightness. Computes a threshold from the image mean by default, "
        "with an optional manual override via the threshold keywoard parameter."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
        "threshold": {
            "type": "number",
            "description": "Optional intensity cutoff (0-255). If omitted or None, the mean image brightness is used."
        }
    },
    "required": ["image"],
    "tags": ["segmentation", "threshold", "binary", "intensity", "foreground"],
    "requires": ["opencv-python-headless", "numpy"],
}


def segment_threshold(image, threshold=None):
    import cv2
    import numpy as np

    # If colour assume BGR
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cutoff = threshold if threshold is not None else image.mean()
    mask = np.where(image > cutoff, 255, 0).astype(np.uint8)
    return mask
