METADATA = {
    "name": "multi_otsu_threshold",
    "description": (
        "Thresholds the image into several different classes using the Otsu algorithm. "
        "Use this when the user wants to isolate more than one class from the background, "
        "or wants to separate 3 or more classes in the image. Creates a multi-class mask "
        "by separating the image into 3 or more classes based on pixel brightness. "
        "Only accepts 8-bit or 16-bit single channel images."
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "classes": {
            "type": "number",
            "description": "The number of classes to split the image into. Defaults to 3. "
            "Yields (classes - 1) threshold values. "
        },
    },
    "required": ["image_data"],
    "tags": ["thresholding", "transform", "feature extraction", "multi-class", "intensity", "foreground"],
    "dependencies": ["scikit-image","numpy","opencv-python-headless"],
    "dependencies": ["scikit-image","numpy","opencv-python-headless"],
}

def multi_otsu_threshold(image_data, classes = 3):
    #     Multi otsu mode:
    #     'src_type == CV_8UC1 || src_type == CV_16UC1'
    #     only accepts single-channel images
    #     outputs CV_8UC1
    
    import skimage as ski
    import cv2
    import numpy as np
    import warnings

    assert classes <= 5, "Too many classes will make the analysis computationally intensive. "
    "If you need to distinguish more than 5 classes, consider a different segmentation approach."

    image = image_data['current']

    assert image.ndim <= 3, "Method only accepts one single-channel image, multi-channel images will be converted to greyscale"
    if image.ndim == 3:
        warnings.warn("Passed multichannel image, converting to greyscale")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype == np.uint16:
        warnings.warn("Converting 16bit to 8bit image")
        image = (image // 256).astype(np.uint8)
    elif image.dtype == np.uint8:
        pass
    else:
        raise TypeError("Only 8 or 16bit images are curently supported") 
    
    thresholds = ski.filters.threshold_multiotsu(image, classes = classes)

    regions = np.digitze(image, bins = thresholds)

    # convert back from int64 (digitize output) to uint8 (useful in pipeline)
    regions = regions / np.max(regions) 
    regions = 255 * regions # Now scale by 255
    regions = regions.astype(np.uint8)

    # return all threshold values
    values = dict(zip(range(classes-1),thresholds))

    image_data['info']['threshold_values'] = values
    image_data['current'] = regions
    return image_data