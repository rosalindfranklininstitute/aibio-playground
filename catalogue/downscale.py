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
}


def downscale(image_data, factor = 2):
    import skimage as ski
    import numpy as np
    import warnings

    image = image_data['current']
    
    assert image.ndim <= 3, "Must pass single or multi-channel image. Dimensions > 3 are not supported"

    # If input is uint8 or uint16, returns an oob float. So convert to float for downscale, 
    # then convert back to original image type for downstream analysis
    type = image.dtype

    if np.issubdtype(type, np.integer) == True:
        image = ski.util.img_as_float(image)
    else:
        raise TypeError("Oops: Unsupported image type. Image should be uint8 or uint16.")

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

    if type == np.uint8:
        downscale = ski.util.img_as_ubyte(downscale)
    elif type == np.uint16:
        downscale = ski.util.img_as_uint(downscale)
    else:
        downscale = ski.util.img_as_uint(downscale)
    
    image_data['current'] = downscale 
    
    return image_data
