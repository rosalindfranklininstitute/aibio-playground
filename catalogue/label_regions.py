METADATA = {
    "name": "label_regions",
    "description": (
        "Computes the labels for a segmented image at the object level. Use this when a user "
        "wants to uniquely label objects for measurement based on connected regions, but doesn't "
        "need to separate overlapping or touching regions of the same class. Returns a labelled "
        "segmentation mask. Input image should already be converted to classes by segmentation "
        "methods such as thresholding. "
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "connectivity": {
            "type": "number",
            "description": "The connecivity to use when considering pixels to be neighbours. "
            "In 2D, connectivity of 1 indicates only orthogonal pixels (cross, 4 neighbours) whereas "
            "2 considers orthogonal and diagonal pixels (8 neighbours). Defaults to 2. "
        },
        "background_value": {
            "type": "number",
            "description": "The integer value of the background class (if any). Setting the "
            "background_value labels all regions with this value as label 0. This means they "
            "they aren't counted during downstream measurement. Defaults to zero."
        },
    },
    "required": ["image_data"],
    "tags": ["segmentation", "feature extraction", "instance", "object", "labels"],
    "dependencies": ["numpy","scikit-image"],
}


def label_regions(image_data, connectivity = 2, background_value = None):
    import skimage as ski
    import warnings

    warnings.warn(message=("Heads up: this function outputs labels. "
    "This should be one of the final steps before measurement."))
    
    image = image_data['current']
    
    if background_value is not None:
        labels = ski.measure.label(image, background = background_value, connectivity = connectivity)
    else:
        labels = ski.measure.label(image, connectivity = connectivity)

    # todo: ski.measure.label returns int64
    image_data['current'] = labels
    return image_data 