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
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
    },
    "required": ["image"],
    "tags": ["invert", "intensity", "foreground", "background"],
    "requires": ["opencv-python-headless"],
}


def invert(image):
    import cv2
    invert = cv2.bitwise_not(image)    
    return invert 