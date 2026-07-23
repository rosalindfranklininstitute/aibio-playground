METADATA = {
    "name": "invert",
    "description": (
        "Inverts an image by returning the complement. For black and white images, this has the "
        "effect of making bright features appear dark, and vice versa. Multi-channel images are "
        "processed per channel. Useful if a dark feature of interest needs to be highlighted to use "
        "techniques like the watershed algorithm, which expects features to be in the foreground, "
        "therefore bright."
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
    "tags": ["invert", "intensity", "foreground", "background"],
    "requires": ["opencv-python-headless"],
}


def invert(image_data):
    import cv2

    image = image_data['current']
    invert = cv2.bitwise_not(image)
    image_data['current'] = invert    
    return image_data 