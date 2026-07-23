METADATA = {
    "name": "watershed",
    "description": (
        "Computes the labels for a binary image using the euclidean distance transform followed by the watershed algorithm. "
        "Uses the local maxima from the distance transform to provide seed coordinates for the watershed algorithm. "
        "Use this when a user wants to uniquely label and measure foreground objects that may be overlapping or touching. "
        "Returns a labelled segmentation mask. Input image must be binary. "
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "footprint": {
            "type": "number",
            "description": "The size of the kernel used for detecting local maxima. "
            "If None, defaults to a 1D kernel. Can impact how overlapping objects are split up."
        },
        "threshold": {
            "type": "number",
            "description": "The relative threshold to use as cutoff when detecting local maxima. "
            "If set, will exlude all maxima below the threshold. "
            "This has the effect of removing smaller objects from the seed coordinates. "
        },
        "show_boundaries": {
            "type": "bool",
            "description": "Whether to show the watershed lines in the labels. If False, no lines shown. "
            "If True, a one-pixel wide line (with label 0) will separate the identified regions."
        },
    },
    "required": ["image_data"],
    "tags": ["segmentation", "distance transform", "watershed", "feature extraction", "binary", "labels", "foreground"],
    "requires": ["scikit-image","scipy","numpy"],
}


def watershed(image_data, footprint = None, threshold = None, show_boundaries = False):
    # This doesn't take in to account different image types yet
    import numpy as np
    import skimage as ski
    from scipy import ndimage as ndi
    import warnings

    warnings.warn(message=("Heads up: the watershed algorithm requires a binary image, and outputs labels. "
    "This should be one of the final steps before measurement."))
    
    image = image_data['current']

    assert len(np.unique(image)) <= 2, "Input image should be binary (2 class)"
    
    distance = ndi.distance_transform_edt(image)
    
    threshold_rel = threshold if threshold is not None else None
    kernel = np.ones((footprint,footprint)) if footprint is not None else np.ones((1))
    
    # find seed coordinates for watershed algorithm
    coords = ski.feature.peak_local_max(distance, footprint = kernel, threshold_rel = threshold_rel, labels = image)

    mask = np.zeros(distance.shape, dtype=bool)
    
    # coordinates are transposed due to difference in indexing for arrays vs points
    mask[tuple(coords.T)] = True

    seeds, _ = ndi.label(mask)

    # because watershed considers valleys (not peaks) we need to invert the distance matrix
    labels = ski.segmentation.watershed(image = -distance, markers = seeds, mask = image, watershed_line = show_boundaries)
    image_data['current'] = labels
    return image_data 