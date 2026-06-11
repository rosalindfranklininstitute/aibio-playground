METADATA = {
    "name": "closing",
    "description": (
        "Closes a binary or greyscale image. "
        "Closing is the same as performing dilation followed by erosion with a kernel (or structuring element) of the same size. "
        "This has the effect of removing small dark objects from the foreground, "
        "without increasing the size of larger foreground objects. "
        "Often used to remove 'salt-and-pepper' noise from images. "
        "Repeated closing with the same size kernel does nothing beyond the first iteration: the operation is idempotent. "
        "Image must be either binary or greyscale, not colour. "
        "Currently implemented with a square kernel only. Kernel size is usually odd, but this is not required. "
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
    },
    "required": ["image"],
    "tags": ["morphological", "transform", "closing", "binary", "feature extraction", "foreground", "despeckle"],
    "requires": ["opencv-python-headless","numpy"],
}

def closing(image, kernel_size = 5):
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    return closing