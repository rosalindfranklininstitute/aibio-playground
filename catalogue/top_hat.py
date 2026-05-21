METADATA = {
    "name": "top_hat",
    "description": (
        "Runs the top-hat transform, also known as the white top-hat. "
        "Use this when a user wants to extract small elements or details from images. "
        "The white tophat locates bright features smaller than the defined kernel size. "
        "Sometimes this transform is used to isolate desired small objects from larger ones. "
        "This can also be used to process images with non-uniform background prior to thresholding "
        "to better distinguish the desired features for downstream segmentation. "
        "White top-hat is the same as subtracting the output of the opening operation from the original image. "
        "Image must be either binary or greyscale, not colour."
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image data. Must be 8bit depth and greyscale to work well."
        },
        "kernel_size": {
            "type": "number",
            "description": "The width and height of the kernel. "
            "Must be positive. Defaults to 5."
        },
    },
    "required": ["image"],
    "tags": ["morphological", "transform", "top-hat", "white top-hat", "binary", "feature extraction", "foreground"],
    "requires": ["opencv-python-headless","numpy"],
}


def top_hat(image, kernel_size = 5):
        # The number of channels can be arbitrary. The depth should be one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
        # So technically this can be done on multichannel images and 32/64 depth
        # But it's not scientifically useful to do so... 
        # Funny things happen if the kernel size is
    import cv2
    import numpy as np
    
    assert kernel_size > 0, "Kernel size must be positive"

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    if image.dtype == np.uint8:
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif image.dtype == np.uint16:
        img = (image // 256).astype(np.uint8)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    else:
        raise TypeError("Image must be 8 or 16bit depth") 
    return tophat