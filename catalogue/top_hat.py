METADATA = {
    "name": "top_hat",
    "description": (
        "Runs the top-hat transform, also known as the white top-hat. "
        "Use this when a user wants to extract small elements or details from images. "
        "The white tophat locates bright features smaller than the defined kernel size. "
        "Sometimes this transform is used to isolate noise, "
        "then the output is subtracted from the original image to remove the unwanted small features. "
        "This can also be used to process images with non-uniform background prior to thresholding "
        "to better distinguish the desired features for downstream segmentation. "
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