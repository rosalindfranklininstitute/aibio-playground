METADATA = {
    "name": "measure_region_properties",
    "description": (
        "Returns region-based properties of a labelled image such as pixel area, "
        "bounding box, centroid coordinate and solidity. Use this when a user wants to "
        "quantify their analysis at the end of the pipeline. Standard properties are returned, "
        "but additional properties can be requested by listing them in the 'extra_properties' "
        "argument. Intensity-based properties of the unprocessed image can be measured using "
        "the labelled image as a mask. Default properties are: "
        "['label','perimeter','area','centroid','num_pixels']"
    ),
    "parameters": {
        "image": {
            "type": "string",
            "description": "Numpy array with image label data. "
        },
        "original": {
            "type": "string",
            "description": "Numpy array with original input image. "
        },
        "extra_properties": {
            "type": "list",
            "description": "List of any additional properties to include. "
            "Must be a property from skimage.measure.regionprops"
        },
        "use_original": {
            "type": "bool",
            "description": "Whether to use the original input image from the user "
            "to measure intensity-based properties. Defaults to False. "
        },
    },
    "required": ["image"],
    "tags": ["measure", "labels", "feature extraction", "shape", "intensity", "size"],
    "requires": ["scikit-image","scipy","numpy"],
}


def measure_region_properties(image, original, extra_properties = [], use_original_image = False):
    # Currently intensity isn't supported as we need to rewrite catalogue 
    # in order to pass the orginial image through the pipeline
    import skimage as ski
    
    original = original if original is not None else image 
    
    defaults = ["label","perimeter","area","centroid","num_pixels"]
    all_props = set.union(set(defaults),set(extra_properties))

    if use_original_image == True:
        props = ski.measure.regionprops_table(label_image = image,
                                              intensity_image = original,
                                              properties = tuple(all_props))  
    else:
        props = ski.measure.regionprops_table(label_image = image,
                                              properties = tuple(all_props))
    return image, original, props 