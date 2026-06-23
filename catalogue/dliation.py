METADATA = {
    "name": "dilation",
    "description": (
        "Dilates a binary or greyscale image. "
        "Each pixel is examined with its neighbours in a given kernel size, "
        "and assigned the highest value in that neighbourhood. "
        "Dilation shrinks dark regions and grows bright ones. "
        "Often used to remove small dark objects before further processing, "
        "for example removing dark background noise. "
        "Image must be either binary or greyscale, not colour. "
        "Currently implemented with a square kernel only."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data. Must be grayscale."
        },
        "kernel_size": {
            "type": "number",
            "description": "The width and height of the kernel. "
            "Must be positive. Defaults to 5."
        },
        "iterations": {
            "type": "number",
            "description": "The number of times the dilation is applied before returning the image."
            "Must be positive. Defaults to 1."
        },
    },
    "required": ["image"],
    "tags": ["morphological", "transform", "dilation", "binary", "feature extraction", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}

def dilation(image, kernel_size = 5, iterations = 1):
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    dilation = cv2.dilate(image,kernel=kernel,iterations=iterations)
    return dilation