METADATA = {
    "name": "simple_threshold",
    "description": (
        "Thresholds the image using a predetermined value. "
        "The threshold is either defined by the user, or computed as the mean image intensity. "
        "Use this when the user wants to isolate objects from the background. "
        "As standard, this method makes bright objects appear white after thresholding, "
        "but setting invert to True will make dark objects appear white instead. "
        "Creates a binary mask by separating the image into foreground and background classes "
        "based on pixel brightness. Only accepts 8-bit or 16-bit single channel images."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
        "threshold": {
            "type": "number",
            "description": "The intensity value used in thresholding. If none, computed from the image mean."
        },
        "invert": {
            "type": "bool",
            "description": "Whether to apply the binary inverse thresholding method (True) or the standard binary (False). Defaults to False."
        },
    },
    "required": ["image"],
    "tags": ["thresholding", "transform", "feature extraction", "binary", "intensity", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}


def simple_thresh(image,threshold = None, invert = False):
    # cvt this to uint 8 is required before setting max val to 255
    # accepts 8, 32 FP, and can be multi-channel 
    # open cv expects the channel to be the last 
    # simple threshold applies the same value to all channels (not that useful)
    import cv2
    import numpy as np

    thresh = threshold if threshold is not None else image.mean()
    if image.dtype == np.uint16:
        maxval = 65535
    elif image.dtype == np.uint8:
        maxval = 255
    
    if invert == False:
        simple_thresh = cv2.threshold(image,thresh=thresh,maxval=maxval, type=cv2.THRESH_BINARY)
    elif invert == True:
        simple_thresh = cv2.threshold(image,thresh=thresh,maxval=maxval, type=cv2.THRESH_BINARY_INV)
    
    return simple_thresh