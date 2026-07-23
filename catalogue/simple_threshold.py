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
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "threshold": {
            "type": "number",
            "description": "The intensity value used in thresholding. If None, computed from the image mean."
        },
        "invert": {
            "type": "bool",
            "description": "Whether to apply the binary inverse thresholding method (True) or the standard binary (False). Defaults to False."
        },
    },
    "required": ["image_data"],
    "tags": ["thresholding", "transform", "feature extraction", "binary", "intensity", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}


def simple_threshold(image_data, threshold = None, invert = False):
    # cvt this to uint 8 is required before setting max val to 255
    # accepts 8, 32 FP, and can be multi-channel 
    # open cv expects the channel to be the last 
    # simple threshold applies the same value to all channels (not that useful)
    import cv2
    import numpy as np

    image = image_data['current']

    thresh = threshold if threshold is not None else image.mean()
    if image.dtype == np.uint16:
        maxval = 65535
    elif image.dtype == np.uint8:
        maxval = 255
    
    if invert == False:
        simple_threshold = cv2.threshold(image,thresh=thresh,maxval=maxval, type=cv2.THRESH_BINARY)
    elif invert == True:
        simple_threshold = cv2.threshold(image,thresh=thresh,maxval=maxval, type=cv2.THRESH_BINARY_INV)
    
    image_data['info']['threshold_value'] = thresh
    image_data['current'] = simple_threshold

    return image_data