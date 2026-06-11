METADATA = {
    "name": "median_filter",
    "description": (
        "Blurs an image using a median filter. Each pixel value is replaced with the median value "
        "within its square neighbourhood, the size of which is defined by the kernel size. "
        "Use this when a user wants to smooth or remove background noise before segmenting an image. "
        "Works especially well with binary data or multi-class labels, "
        "as no intermediate pixel values are calculated or used. "
        "The kernel size is defined in pixels and must be positive and odd."
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
    },
    "required": ["image"],
    "tags": ["smoothing", "denoise", "blur", "median", "filter", "pre-processing", "binary"],
    "requires": ["opencv-python-headless"],
}


def median_filter(image, kernel_size = 5):
    import cv2

    assert kernel_size % 2 != 0, "Kernel size must be odd"
    assert kernel_size > 0, "Kernel size must be positive"

    blurred_image = cv2.medianBlur(image,kernel_size)
    return blurred_image