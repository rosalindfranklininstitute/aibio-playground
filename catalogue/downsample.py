METADATA = {
    "name": "downsample",
    "description": (
        "Downsamples an image by first blurring it with a gaussian kernel, "
        "then removing even rows and columns. Each iteration reduces the size "
        "of the image by a factor of 2. Use this if downscale.py doesn't give "
        "the desired result. This method builds an image pyramid, so may produce "
        "large variables during processing. "
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "factor": {
            "type": "number",
            "description": "The factor to reduce the image by. Must be a power series "
            "of 2 (2, 4, 8 16 etc). Defaults to 2."
        },
    },
    "required": ["image_data"],
    "tags": ["downsample", "downscale", "resize", "rescale", "image pyramid"],
    "requires": ["opencv-python-headless"],
}


def downsample(image_data, factor = 2):
    import cv2
    import math
    image = image_data['current']

    iterations = math.log2(factor)
    assert iterations % 1 == 0, "Factor must be 2^n"

    pyramid = [image]
    for i in range(int(iterations)): 
        image = cv2.pyrDown(image)
        pyramid.append(image)

    image_data['current'] = pyramid[-1]    
    return image_data