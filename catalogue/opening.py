METADATA = {
    "name": "opening",
    "description": (
        "Opens a binary or greyscale image. "
        "Opening is the same as performing erosion followed by dilation with a kernel (or structuring element) of the same size. "
        "This has the effect of removing small bright objects from the background, "
        "without reducing the size of larger foreground objects. "
        "Often used to remove 'salt-and-pepper' noise from images. "
        "Repeated opening with the same size kernel does nothing beyond the first iteration: the operation is idempotent. "
        "Image must be either binary or greyscale, not colour. "
        "Currently implemented with a square kernel only. Kernel size is usually odd, but this is not required. "
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
    "required": ["image_data"],
    "tags": ["morphological", "transform", "opening", "binary", "feature extraction", "background", "despeckle"],
    "requires": ["opencv-python-headless","numpy"],
}

def opening(image_data, kernel_size = 5):
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"
    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    image = image_data['current']

    opening = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    image_data['current'] = opening
    return image_data