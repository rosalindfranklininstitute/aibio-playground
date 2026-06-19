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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ###The neat case: funcs defined with legible args, uses partial and reduce from functools with actual params supplied with nested **defaults
    """)
    return


@app.cell
def _():
    mo.md("""
    This notebook uses a sortable list to chain functions together and then apply them.

    The default parameters are generated from the list of functions and saved in dict 'defaults' Currently the dictionary includes all image analysis functions in the environment, rather than those selected in the list.

    The dictionary can be altered but currently isn't editable in app view. Using a mo.ui.array or mo.ui.dictionary coerces all values to strings, which is not useful.
    """)
    return


@app.cell
def _():
    seeded = cv.imread("Seeded-Thresh-M.tif",cv.IMREAD_UNCHANGED)
    scaffold = cv.imread("Scaffold-Thresh-M.tif",cv.IMREAD_UNCHANGED)
    # These paths are for the VM

    selector = mo.ui.dropdown(options=["seeded","scaffold","mitosis","coins","blobs"])
    mo.vstack([mo.md("Select the image you want to process"),selector])
    return scaffold, seeded, selector


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
    elif selector.value == "blobs":
        blobs = ski.data.binary_blobs()
        image = np.uint8(np.where(blobs==True,255,0))
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
        erode = cv.erode(image,kernel,iterations=iterations)
        return erode

    def dilate(image, kernel_size = 5,iterations = 1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        dilate = cv.dilate(image,kernel,iterations=iterations)
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
    funcs = {k:v for k,v in locals().items() if callable(v) and v.__module__==__name__ and not k.startswith('_') and not k.endswith("funcs") and not k.startswith("get")}

    current_function_names = funcs.keys()
    return current_function_names, funcs


@app.cell
def _():
    reload_funcs = mo.ui.button(label="reload available functions")
    reload_funcs
    # Functions have to be reloaded if they are altered because mutations don't trigger rerunning cells.
    return (reload_funcs,)


@app.cell
def _(defaults, funcs, get_funcs, image, widget):
    def make_partial_funcs(funcs,**kwargs):
        """
        Creates partials (func objects) from all funcs given.
        This means we don't need to pass **kwargs to every function in the chaining,
        which isn't possible when using reduce
        """
        from functools import partial
        return [partial(func,**kwargs[func.__name__]) for func in funcs]

    def reduce_funcs(sequence, image, **kwargs):
        """
        Reduce list of functions to a single function by iterating over the list
        """
        from functools import reduce
        composed = reduce(lambda im, func: func(im),sequence,image)
        return composed

    def map_funcs_kw(image, func_list, current_functions, **kwargs):
        """
        Combines function retrieval, making partials, and reducing the partials to a single composed output.
        Returns composed: the fully processed image, and funcs: the list of funcs applied.
        """
        funcs = get_funcs(func_list,current_functions)
        sequence = make_partial_funcs(funcs,**kwargs)
        composed = reduce_funcs(sequence,image,**kwargs)
        return funcs, composed

    _funcs, _composed = map_funcs_kw(image,widget.value.get("value"),funcs, **defaults)
    mo.image_compare(image,_composed,width=450)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # dictionary = ({
    #     "gaussian_blur":{"kernel_size":15,"sigma":0},
    #     "otsu_thresh":{},
    #     "median_filter":{"kernel_size":3},
    #     "erode":{"kernel_size":3,"iterations":1}
    # })
    return


@app.cell
def _():
    def get_funcs(func_list,current_functions):
        """
        Get functions from a specified list (func_list), from the dictionary of all funcs in the environment (current_functions)
        """
        funcs = []
        for item in func_list:
            next = current_functions.get(item)
            print(next)
            funcs.append(next)
        return funcs

    def get_defaults(fn):
        """
        Get the default values of the passed function or method.
        """
        output = {}
        if fn.__defaults__ is not None:
            # Get the names of all provided default values for args
            default_varnames = list(fn.__code__.co_varnames)[:fn.__code__.co_argcount][-len(fn.__defaults__):]
            # Update the output dictionary with the default values
            output.update(dict(zip(default_varnames, fn.__defaults__)))
        if fn.__kwdefaults__ is not None:
            # Update the output dictionary with the keyword default values
            output.update(fn.__kwdefaults__)
        return output

    def get_params(func_list,funcs):
        """
        Collect all defaults for a list of functions and return as a nested params dict.
        Func_list = fn names as str
        funcs = dict of fn name : fn
        """
        params = dict()
        for fn in func_list:
            params.update({fn : get_defaults(funcs[fn])})
        return params

    return get_funcs, get_params


@app.cell
def _(current_function_names, funcs, get_params):
    defaults = get_params(current_function_names,funcs)
    return (defaults,)


@app.cell(hide_code=True)
def _(defaults):
    mo.tree(defaults)
    return


@app.cell
def _():
    # initial_code = """# implement foo below
    # dictionary = ({
    #     "gaussian_blur":{"kernel_size":15,"sigma":0},
    #     "otsu_thresh":{},
    #     "median_filter":{"kernel_size":3},
    #     "erode":{"kernel_size":3,"iterations":1}
    # })
    # """
    # foo = "dictionary = "+defaults.items()

    # editor = mo.ui.code_editor(value=foo, language="python")
    # editor
    return


@app.cell
def _():
    simple_dict = {"foo":True,"bar":5,"blah":"string"}
    tmp = mo.ui.dictionary(
        {
            f"{k}": mo.ui.text(value=str(v))
            for k,v in simple_dict.items()
        },label="gaussian_blur"
    )
    tmp
    # Only works for flat dicts
    return simple_dict, tmp


@app.cell
def _():
    mo.md("""
    UI text entry coerces all entries to string:
    """)
    return


@app.cell
def _(simple_dict):
    simple_dict
    return


@app.cell
def _(tmp):
    tmp.value
    return


@app.cell
def _():
    return


@app.cell
def _(simple_dict):
    mo.tree(simple_dict)
    return


@app.cell
def _():
    run_button = mo.ui.run_button(label="click to update parameters")
    return (run_button,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _():
    # mo.stop(not run_button.value, mo.md("Press the button to submit parameters."))
    # namespace = {}
    # # Exec what's in the editor box - allows user to modify function!
    # exec(editor.value, namespace)

    # dictionary = namespace["dictionary"]
    return


@app.cell
def _():
    # used_funcs, composed_image = map_funcs(image, widget.value.get("value"),funcs)
    # used_funcs
    # mo.image(composed_image)
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


if __name__ == "__main__":
    app.run()
