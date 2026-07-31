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
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "kernel_size": {
            "type": "number",
            "description": "The width and height of the smoothing kernel. "
            "Must be positive and odd. Defaults to 5."
        },
    },
    "required": ["image_data"],
    "tags": ["smoothing", "denoise", "blur", "median", "filter", "pre-processing", "binary"],
    "dependencies": ["opencv-python-headless"],
}


def median_filter(image_data, kernel_size = 5):
    import cv2

    assert kernel_size % 2 != 0, "Kernel size must be odd"
    assert kernel_size > 0, "Kernel size must be positive"

    image = image_data['current']

    blurred_image = cv2.medianBlur(image,kernel_size)
    image_data['current'] = blurred_image
    return image_data
