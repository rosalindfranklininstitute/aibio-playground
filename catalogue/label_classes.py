METADATA = {
    "name": "label_classes",
    "description": (
        "Computes the labels for a binary or multi-class image at the class level. "
        "Use this when a user wants to uniquely label classes for measurement without generating "
        "instances based on connected regions. Returns a labelled segmentation mask. Also allows "
        "measurement of regions where no class is considered background. Input image should already "
        "be converted to classes by segmentation methods such as thresholding."
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "background_class": {
            "type": "boolean",
            "description": "Whether or not the image for labelling contains a background class. "
            "If True, the lowest value in the current image is considered background, and labelled "
            "with value 0. If False, labelling starts at 1. Defaults to True. "
        },
    },
    "required": ["image_data"],
    "tags": ["segmentation", "feature extraction", "binary", "multi-class", "labels"],
    "dependencies": ["numpy"],
}


def label_classes(image_data, background_class = True):
    import numpy as np
    import warnings

    warnings.warn(message=("Heads up: this function outputs labels. "
    "This should be one of the final steps before measurement."))
    
    image = image_data['current']
        
    values = np.unique(image)
    assert len(values) <= 20, "Image has more than 20 unique values to assign labels, "
    "have you used a segmentation technique such as thresholding yet?"

    labels = np.digitize(image,values)
    
    if background_class == True:
        labels -=1
    elif background_class == False:
        pass

    # todo: np.digitize returns int64
    image_data['current'] = labels
    return image_data 
