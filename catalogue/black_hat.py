METADATA = {
    "name": "black_hat",
    "description": (
        "Runs the black hat transform, also known as the black top-hat "
        "Use this when a user wants to extract small elements or details from images. "
        "The black hat locates dark features smaller than the defined kernel size. "
        "Located pixels will be highlighted in the returned image, so will appear bright. "
        "Sometimes this transform is used to find the dark edges of objects. "
        "Black hat is the same as subtracting the original image from the output of the closing operation. "
        "Image must be either binary or greyscale, not colour."
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "kernel_size": {
            "type": "number",
            "description": "The width and height of the kernel. "
            "Must be positive. Defaults to 5."
        },
    },
    "required": ["imag_data"],
    "tags": ["morphological", "transform", "black hat", "top-hat", "binary", "feature extraction", "foreground"],
    "dependencies": ["opencv-python-headless","numpy"],
    "dependencies": ["opencv-python-headless","numpy"],
}


def black_hat(image_data, kernel_size = 5):
        # The number of channels can be arbitrary. The depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
        # So technically this can be done on multichannel images and 32/64 depth
        # But it's not scientifically useful to do so... 
        # Funny things happen if the kernel size is even.
    
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    image = image_data['current']

    if image.dtype == np.uint8:
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKPHAT, kernel)
    elif image.dtype == np.uint16:
        img = (image // 256).astype(np.uint8)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise TypeError("Image must be 8 or 16bit depth") 
    
    image_data['current'] = blackhat
    return image_data