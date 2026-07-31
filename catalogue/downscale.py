METADATA = {
    "name": "downscale",
    "description": (
        "Downscales an image by calculating the local mean for pixel neigbourhoods "
        "of size equal to the factor parameter. If a non-integer factor is supplied, "
        "the method will use area-based interpolation instead. Use this when a user "
        "wants to speed up processing by reducing the image size. An alternative method "
        "from the open-cv library is downsample.py"
    ),
    "parameters": {
        "image_data": {
            "type": "dict",
            "description": "Dictionary containing original image ('source', Numpy array), "
            "latest image to be processed ('current', Numpy array), and any data from current/prior "
            "processing steps ('info', dict)"
        },
        "factor": {
            "type": "number",
            "description": "The scale factor to downscale the image by. "
            "A scale factor of 2 will reduce each image dimension by half. "
            "Factors less than 1 will increase the size of the image. "
        },
    },
    "required": ["image_data"],
    "tags": ["downsample", "downscale", "resize", "rescale", "interpolation"],
    "dependencies": ["scikit-image"],
    "dependencies": ["scikit-image"],
}


def downscale(image_data, factor = 2):
    import skimage as ski

    image = image_data['current']
    
    assert image.ndim <= 3, "Must pass single or multi-channel image. Dimensions > 3 are not supported"

    w, h, *c = image.shape
    
    if factor % 1 == 0:
        if image.ndim == 3:
            downscale = ski.transform.downscale_local_mean(image, (factor,factor,1))
        elif image.ndim == 2:
            downscale = ski.transform.downscale_local_mean(image, (factor,factor))
    else:  
        newscale = (int(w/factor), int(h/factor), *c)
        if image.ndim == 3:
            downscale = ski.transform.resize_local_mean(image,output_shape = newscale, preserve_range = True, channel_axis = 2)
        elif image.ndim == 2:
            downscale = ski.transform.resize_local_mean(image, output_shape = newscale, preserve_range = True)
    
    image_data['current'] = downscale 
    
    return image_data