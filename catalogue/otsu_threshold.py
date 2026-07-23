METADATA = {
    "name": "otsu_threshold",
    "description": (
        "Thresholds the image using the optimal value computed by the Otsu algorithm. "
        "Use this when the user wants to isolate objects from the background. "
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
    },
    "required": ["image_data"],
    "tags": ["thresholding", "transform", "feature extraction", "binary", "intensity", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}

def otsu_threshold(image_data):
    #     THRESH_OTSU mode:
    #     'src_type == CV_8UC1 || src_type == CV_16UC1'
    #     only accepts single-channel images
    import cv2
    import numpy as np
    import warnings

    image = image_data['current']

    assert image.ndim <= 3, "Method only accepts one single-channel image, multi-channel images will be converted to greyscale"
    if image.ndim == 3:
        warnings.warn("Passed multichannel image, converting to greyscale")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype == np.uint16:
        maxval = 65535
    elif image.dtype == np.uint8:
        maxval = 255
    
    val, otsu_threshold = cv2.threshold(image, thresh = 0, maxval = maxval, type = cv2.THRESH_OTSU)
    
    image_data['info']['threshold_value'] = val
    image_data['current'] = otsu_threshold
    return image_data
