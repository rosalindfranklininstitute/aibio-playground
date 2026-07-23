METADATA = {
    "name": "morphological_gradient",
    "description": (
        "The morphological gradient is a technique to highlight sharp contrasts in intensity and object boundaries. "
        "This operation is the same as subtracting the erosion output from the dilation output "
        "of a binary or greyscale image, using a kernel (or structuring element) of the same size. "
        "This has the effect of highlighting the edges of objects or features.  "
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
    "tags": ["morphological", "transform", "morphological gradient", "binary", "feature extraction", "edges", "edge detection"],
    "requires": ["opencv-python-headless","numpy"],
}

def morphological_gradient(image_data, kernel_size = 5):
    import cv2
    import numpy as np

    assert kernel_size > 0, "Kernel size must be positive"
    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    image = image_data['current']

    morphological_gradient = cv2.morphologyEx(image,cv2.MORPH_GRADIENT,kernel)
    image_data['current'] = morphological_gradient
    return image_data