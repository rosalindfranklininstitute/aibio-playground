METADATA = {
    "name": "erosion",
    "description": (
        "Erodes a binary or greyscale image. "
        "Each pixel is examined with its neighbours in a given kernel size, "
        "and assigned the lowest value in that neighbourhood. "
        "Erosion grows dark regions and shrinks bright ones."
        "Often used to remove small bright objects before further processing,"
        "for example removing bright background noise."
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
            "description": "The number of times the erosion is applied before returning the image."
            "Must be positive. Defaults to 1."
        },
    },
    "required": ["image"],
    "tags": ["morphological", "transform", "erosion", "binary", "feature extraction", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}

def erosion(image, kernel_size = 5, iterations = 1):
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    erosion = cv2.erode(image,kernel=kernel,iterations=iterations)
    return erosion