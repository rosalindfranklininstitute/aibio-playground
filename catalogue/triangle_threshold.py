METADATA = {
    "name": "triangle_threshold",
    "description": (
        "Thresholds the image using the optimal value computed by the Triangle algorithm. "
        "Use this when the user wants to isolate objects from the background. "
        "Creates a binary mask by separating the image into foreground and background classes "
        "based on pixel brightness. Only accepts 8-bit single channel images."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
    },
    "required": ["image"],
    "tags": ["thresholding", "transform", "feature extraction", "binary", "intensity", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}

def triangle_thresh(image):
    #     THRESH_TRIANG:E mode:
    #     'src_type == CV_8UC1'
    #     only accepts single-channel images
    import cv2
    import numpy as np

    assert image.ndim <= 3, "Method only accepts one single-channel image, multi-channel images will be converted to greyscale"
    if image.ndim == 3:
        # with mo.redirect_stdout():
        #     print("Warning: passed multichannel image, converting to greyscale")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype == np.uint16:
        # with mo.redirect_stdout():
        #     print("Warning: passed 16bit image, converting to 8bit for processing")
        image = (image // 256).astype(np.uint8)
        maxval = 255
    elif image.dtype == np.uint8:
        maxval = 255
    
    
    _, triangle_thresh = cv2.threshold(image, thresh = 0, maxval = maxval, type = cv2.THRESH_OTSU)
    
    return triangle_thresh

# Returns only second argument = thresholded image. Threshold value _ not currently returned