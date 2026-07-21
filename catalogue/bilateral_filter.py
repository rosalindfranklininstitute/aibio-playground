METADATA = {
    "name": "bilateral_filter",
    "description": (
        "Blurs an image using a bilateral filter: an edge-preserving filter that averages pixels based on "
        "their spatial closeness and radiometric similarity. Use this when a user wants to smooth an image "
        "without losing edges to blurring. The (radial) distance defines the neighbourhood for sampling. "
        "It can be defined in pixels (must be odd), or can be inferred from the standard deviation "
        "of the space instead (distance is one third the sigma space). The filter takes account of nearby "
        "pixels (sigma space) and pixels with close colour intensities (sigma colour). "
        "Only takes 8U and 32F image types. Can be single or multi-channel. "
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data"
        },
        "distance": {
            "type": "number",
            "description": "The radial distance of the neighbourhood considered for filtering. "
            "If negative, or blank, defaults to 1/3 of sigma_space. Otherwise must be positive and odd."
        },
        "sigma_colour":{
            "type": "number",
            "description": "the standard deviation or sigma for the colour filter."
            "Defaults to 50."
        },
        "sigma_space":{
            "type": "number",
            "description": "the standard deviation or sigma for the space filter."
            "Defaults to 5."
        }
    },
    "required": ["image"],
    "tags": ["smoothing", "denoise", "blur", "filter", "pre-processing"],
    "requires": ["opencv-python-headless"],
}


def bilateral_filter(image, distance = None, sigma_colour = 50, sigma_space = 5):
    # Bilateral filtering is only implemented for CV_8U and CV_32F images
    # Currently only implemented conversion from CV_16U to CV_8U, FP not supported. 
    import cv2
    import numpy as np
    
    assert sigma_colour >= sigma_space, "Are you sure? If colour and space variations are similar, maybe try a gaussian blur instead"
    assert sigma_space <= 100, "Large values for sigma space will make the analysis computationally intensive"

    dist = distance if distance is not None else sigma_space//3

    assert dist <= 9 and dist > 0, "Distance must be positive. Larger values of distance will make the analysis very slow. Recommended value = 5"
    
    if image.dtype == np.uint8:
        blurred_image = cv2.bilateralFilter(image, dist, sigma_colour, sigma_space)
    elif image.dtype == np.uint16:
        
        img = (image // 256).astype(np.uint8)
        blurred_image = cv2.bilateralFilter(img, dist, sigma_colour, sigma_space)
    else:
        raise TypeError("Image must be 8 or 16bit depth") 

    blurred_image = cv2.bilateralFilter(image, dist, sigma_colour, sigma_space)
    return blurred_image
