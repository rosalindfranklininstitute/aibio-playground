import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from wigglystuff import SortableList
    import numpy as np
    import cv2 as cv
    import os
    import skimage as ski
    import inspect


@app.cell
def _():
    # avail_funcs = mo.ui.multiselect(
    #     options=current_function_names, label="choose some options"
    # )
    # avail_funcs
    return


@app.cell
def _(current_function_names):
    avail_funcs = mo.ui.anywidget(
        SortableList(
            current_function_names,
            editable=False,
            addable=False,
            removable=False,
            label="Available Tools"
        )
    )
    avail_funcs
    return (avail_funcs,)


@app.cell
def _(avail_funcs, reload_funcs):
    reload_funcs
    widget = mo.ui.anywidget(
        SortableList(
            avail_funcs.value.get("value"),
            editable=True,
            addable=True,
            removable=True,
            label="My Sortable List of Tools"
        )

    )
    widget
    return (widget,)


@app.cell
def _():
    seeded = cv.imread("marimo-testing/Seeded-Thresh-M.tif",cv.IMREAD_UNCHANGED)
    scaffold = cv.imread("marimo-testing/Scaffold-Thresh-M.tif",cv.IMREAD_UNCHANGED)


    selector = mo.ui.dropdown(options=["seeded","scaffold","mitosis","coins"])
    mo.vstack([mo.md("Select the image you want to threshold"),selector])
    return scaffold, seeded, selector


@app.cell
def _(scaffold, seeded, selector):
    if selector.value == "seeded":
        image = seeded
        maxval=np.max(image)
    elif selector.value == "scaffold":
        image = scaffold
        maxval=np.max(image)
    elif selector.value == "mitosis":
        image = ski.data.human_mitosis()
        maxval=np.max(image)
    elif selector.value == "coins":
        image = ski.data.coins()
        maxval=np.max(image)
    else:
        image = ski.data.hubble_deep_field()
        # image = cv.imread("marimo-testing/example-image-blobs.jpg", cv.IMREAD_GRAYSCALE)
        maxval=np.max(image)
    return (image,)


@app.cell
def _():
    def otsu_thresh(image):
        # cvt this to uint 8 is required before setting max val to 255
        #     THRESH_OTSU mode:
        #     'src_type == CV_8UC1 || src_type == CV_16UC1'
        # only accepts single-channel images

        assert image.ndim <= 3, "Method only accepts one single-channel image, multi-channel images will be converted to greyscale"
        if image.ndim == 3:
            with mo.redirect_stdout():
                print("Warning: passed multichannel image, converting to greyscale")
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        if image.dtype == np.uint16:
            maxval = 65535
        elif image.dtype == np.uint8:
            maxval = 255
        _, otsu_thresh = cv.threshold(image, thresh = 0, maxval=maxval, type=cv.THRESH_OTSU)
        return otsu_thresh

    def gaussian_blur(image, kernel_size = 5,sigma = 0):
        # import cv2
        # import numpy as np
        assert kernel_size % 2 != 0, "Kernel size must be odd"
        assert kernel_size > 0, "Kernel size must be positive"
        blurred_image = cv.GaussianBlur(image,(kernel_size,kernel_size),sigma)
        return blurred_image

    def median_filter(image,kernel_size = 5):
        # import cv2
        assert kernel_size % 2 != 0, "Kernel size must be odd"
        assert kernel_size > 0, "Kernel size must be positive"
        blurred_image = cv.medianBlur(image,kernel_size)
        return blurred_image

    def bilateral_filter(image, kernel_size = 5, sigma_colour = 15, sigma_space = 4):
        # import cv2
        # assert kernel_size > 0, "Kernel size must be positive"
        blurred_image = cv.bilateralFilter(image, kernel_size, sigma_colour, sigma_space)
        return blurred_image



    def erode(image, kernel_size = 5, iterations = 1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        erode = cv.erode(image,kernel,iterations)
        return erode

    def dilate(image, kernel_size = 5,iterations = 1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        dilate = cv.dilate(image,kernel,iterations)
        return dilate

    def tophat(image,kernel_size = 5):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        tophat = cv.morphologyEx(image,cv.MORPH_TOPHAT,kernel)
        return tophat

    def blackhat(image,kernel_size = 5,iterations = 1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        blackhat = cv.morphologyEx(image,cv.MORPH_BLACKHAT,kernel)
        return blackhat

    def opening(image,kernel_size = 5):
        #opens - erosion followed by dilation
        #removes bright noise in background
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        opening = cv.morphologyEx(image,cv.MORPH_OPEN,kernel)
        return opening

    def closing(image,kernel_size = 5):
        #closes - dilation followed by erosion
        #closes small (dark) holes in foreground (bright)
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        closing = cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
        return closing

    return


@app.cell
def _(reload_funcs):
    reload_funcs
    funcs = {k:v for k,v in locals().items() if callable(v) and v.__module__==__name__ and not k.startswith('_') and not k.endswith("funcs")}

    current_function_names = funcs.keys()
    return current_function_names, funcs


@app.cell
def _():
    reload_funcs = mo.ui.button(label="reload available functions")
    reload_funcs
    return (reload_funcs,)


@app.cell
def _():
    # func_selector = mo.ui.dropdown(current_function_names)
    # mo.vstack([mo.md("Select the function to apply"),func_selector],justify="start")
    return


@app.cell
def _():
    # kernel = np.ones((kernel_size.value,kernel_size.value),np.uint8)
    # kernel
    return


@app.cell
def _():
    # kernel_size = mo.ui.number(value=3,start=1,step=2)
    # kernel_size
    return


@app.cell
def _():
    # def nesting(image, *args):

    return


@app.function
def get_funcs(func_list,current_functions):
    funcs = []
    for item in func_list:
        next = current_functions.get(item)
        print(next)
        funcs.append(next)
    return funcs


@app.function
def map_funcs(obj, func_list, current_functions):
    # return [func(obj) for func in func_list]
    from functools import reduce
    funcs = get_funcs(func_list,current_functions)
    # I was under the impression that the OP wanted to compose the functions,
    # i.e. f3(f2(f1(f0(obj))), for which the line below is applicable:
    composed = reduce(lambda o, func: func(o), funcs, obj)
    return funcs, composed


@app.cell
def _():
    mo.md("""
    Right now, only using default values because each function is only applied to the array. No idea whether it would work with json

    I haven't implemented it, but I think we could create a list of all suggested functions (avail functions here) as a check box exercise. We could use the checkboxes to create the sortable list, to allow the functions to be called in different orders.

    If we made it so that each selected function had an editable source code, we could change the default params of the function. This isn't the same as generating the json to have a dictionary that passes all parameters to all functions though. We COULD have a method where the listed functions are shown, the dictionary is generated, and the dictionary can be edited to rerun with different parameters.
    """)
    return


@app.cell
def _(image):
    dictionary = ({"image":image,
        "gaussian_blur":{"kernel_size":5,"sigma":0}
    })
    return (dictionary,)


@app.cell(hide_code=True)
def _(dictionary):
    mo.tree(dictionary)
    return


@app.cell
def _(funcs, image, widget):
    used_funcs, composed_image = map_funcs(image, widget.value.get("value"),funcs)
    used_funcs
    mo.image(composed_image)
    return


@app.cell
def _():
    # for i in range(len(widget.value.get("value"))):
    #     func = funcs.get(widget.value.get("value")[i])
    #     spec = inspect.getfullargspec(func)
    #     args = spec.args
    #     print(spec)
    #     if "kernel" in args:
    #         output = func(image, kernel)
    #     else:
    #         output = func(image)
    # mo.image(output)
    return


@app.cell
def _():
    # reload_funcs
    # message = inspect.getsource(func)
    # mo.vstack([mo.md(f"You selected function **'{func.__name__}'** with source code:"),mo.plain_text(message)])
    return


@app.cell
def _(image):
    mo.image(image)
    return


@app.cell
def _():
    # mo.image_compare(image,output,width=image.shape[1])
    return


if __name__ == "__main__":
    app.run()
