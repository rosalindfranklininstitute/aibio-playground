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
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "extra_properties": {
            "type": "list",
            "description": "List of any additional properties to include. "
            "Must be a property from skimage.measure.regionprops"
        },
    },
    "required": ["image_data"],
    "tags": ["measure", "labels", "feature extraction", "shape", "intensity", "size"],
    "dependencies": ["scikit-image","scipy","numpy"],
    "dependencies": ["scikit-image","scipy","numpy"],
}


def measure_region_properties(image_data, extra_properties = []):
    import skimage as ski

    defaults = ["label","perimeter","area","centroid","num_pixels"]
    all_props = set.union(set(defaults),set(extra_properties))

    props = ski.measure.regionprops_table(label_image = image_data['current'],
                                          intensity_image = image_data['source'],
                                          properties = tuple(all_props))  

    image_data['info']['measurements'] = props
    return image_data