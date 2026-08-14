import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex
from matplotlib.cm import ScalarMappable, prism
from matplotlib.gridspec import GridSpec
from skimage import exposure



def color_cell(rowId, columnName, value):
    """
    Marimo style callable for mo.ui.table:
    Styles the cells in the 'color' column of measurements to be highlighted 
    according to the hex string (corresponding to the label colour in the image)
    """
    if columnName ==  "color":
        hex = str(rgb2hex(value))
        return {
            "backgroundColor":hex
        }
    else:
        return {}

def color_labels(arr: np.ndarray) -> np.ndarray:
    """
    Convert a labelled image to a RGBA colormap image for visualisation.
    Normalises the colormap ('prism') to span the range of labels. 
    Background class (label 0) defaults to black. This complements get_hexvals()
    and will produce matching colors for each label.
    """
    ma = np.ma.masked_equal(arr,0)
    range = Normalize(np.min(arr),np.max(arr), clip = False)
    cmap = ScalarMappable(range, cmap = prism)
    out = cmap.to_rgba(arr)
    mask = ma.mask
    # out[mask]=np.array((0,0,0,0)) # make bkg class transparent
    out[mask] = np.array((0,0,0,1)) # make bkg class black
    return out

def get_hexvals(image_data: dict):
    """
    Get label info from any label function, and use the range of values
    to build a colormap for visualisation. This complements color_labels()
    and will produce matching colors for each label.
    """
    # We expect that label tools will return label metadata to use for this purpose
    if image_data.get('info').get('labels'):
        n = image_data['info']['labels']['num_labels']
        cmap = plt.get_cmap('prism',n)
        image_data['info']['labels']['cmap'] = cmap

        if image_data.get('info').get('measurements'):
            labels = image_data['info']['measurements']['label']
            colors = np.array([rgb2hex(cmap(item)) for item in labels])

        image_data['info']['measurements']['color'] = colors
        return image_data

    # But if there isn't label metadata, we can guess from measurement data
    else:
        if image_data.get('info').get('measurements'):
                n = image_data['info']['measurements']['label'][-1]
                m = image_data['info']['measurements']['label'][0]
                if m == 0:
                    cmap = plt.get_cmap('prism',n+1)
                else:
                    cmap = plt.get_cmap('prism',n)
                labels = image_data['info']['measurements']['label']
                colors = np.array([rgb2hex(cmap(item)) for item in labels])
                image_data['info']['measurements']['color'] = colors
                return image_data

        # If no label or measurement data, we can't create label colors
        else:
            return image_data


def build_image_histogram(arr: np.ndarray):
    # If only 2D array, histogram is greyscale/single channel 
    plt.rcParams.update({'font.size':5}) # This means the fig size and font are well matched
    if arr.ndim == 2:
        h, hc = exposure.histogram(arr)
        fig, ax = plt.subplots()
        ax.plot(hc, h, lw=1, c="black")
        ax.set_title('Single channel image')
        plot = plt.gca()
        fig.set_size_inches(4,3)
        return plot
    
    # If 3 channels, we assume that they are rgb, and plot with these colors
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        h, hc = exposure.histogram(arr)
        r, rc = exposure.histogram(arr[:,:,0])
        g, gc = exposure.histogram(arr[:,:,1])
        b, bc = exposure.histogram(arr[:,:,2])

        fig = plt.figure(layout="constrained")

        gs = GridSpec(3,3, figure= fig)
        grey = fig.add_subplot(gs[:,:2])
        red = fig.add_subplot(gs[0,2:])
        green = fig.add_subplot(gs[1,2:])
        blue = fig.add_subplot(gs[2,2:])
        
        grey.plot(hc, h, lw=1, c="black")
        grey.get_yaxis().set_visible(False)
        grey.set_title('All channels')
        
        red.plot(rc, r, lw=1, c="red")
        red.get_yaxis().set_visible(False)
        red.get_xaxis().set_visible(False)
        red.set_title('Single channels (RGB)')
        
        green.plot(gc, g, lw=1, c="green")
        green.get_yaxis().set_visible(False)
        green.get_xaxis().set_visible(False)
        # green.set_title('Green channel')
        
        blue.plot(bc, b, lw=1, c= "blue")
        blue.get_yaxis().set_visible(False)
        # blue.set_title('Blue channel')

        plot = plt.gcf()
        fig.set_size_inches(4,3)
        return plot
    
    # If not 3 channels, we can't assume that they are rgb, but we can plot hists for all of them
    # But this means that RGBA channels are plotted with 4 histograms
    # Lots of channels will squish the shape poorly
    else:
        h, hc = exposure.histogram(arr)
        num_channels = arr.shape[-1]
        single_channels = [exposure.histogram(arr[:,:,i]) for i in range(num_channels)]

        fig = plt.figure(layout="constrained")
        gs = GridSpec(num_channels,num_channels, figure=fig)

        grey = fig.add_subplot(gs[:,:num_channels-1])
        grey.plot(hc, h, lw=1,c="black")
        grey.set_title("All channels")
        
        for i in range(num_channels):
            ax = fig.add_subplot(gs[i,num_channels-1:])
            ax.plot(single_channels[i][1],single_channels[i][0],lw=1, c="grey")
            
            if i == 0:
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.set_title('Single channels')
            elif i == num_channels-1:
                ax.get_yaxis().set_visible(False)
            else:
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
        
        plot = plt.gcf()
        fig.set_size_inches(4,3)
        return plot