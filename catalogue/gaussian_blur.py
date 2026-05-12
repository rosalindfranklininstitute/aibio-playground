METADATA = {
    "name": "gaussian_blur",
    "description": (
        "Blurs an image using a gaussian filter. "
        "Use this when a user wants to remove background noise before segmenting the image. "
        "The kernel size can be defined in pixels (must be odd), "
        "or the standard deviation (sigma) of the gaussian curve can be used instead. "
        "For images with multiple channels, the blur is computed for each channel separately."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
        "kernel_size": {
            "type": "number",
            "description": "The width and height of the smoothing kernel. "
            "Must be positive and odd. Defaults to 5."
        },
        "sigma":{
            "type": "number",
            "decription": "the standard deviation or sigma of the smoothing kernel."
            "Defaults to 0."
        }
    },
    "required": ["image"],
    "tags": ["smoothing", "denoise", "blur", "pre-processing"],
    "requires": ["opencv-python-headless"],
}


def gaussian_blur(image, kernel_size = 5, sigma = 0):
    import cv2

    assert kernel_size % 2 != 0, "Kernel size must be odd"
    assert kernel_size > 0, "Kernel size must be positive"

    blurred_image = cv2.GaussianBlur(image,(kernel_size,kernel_size),sigma)
    return blurred_image