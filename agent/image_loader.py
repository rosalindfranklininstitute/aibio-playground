import io, sys, os, base64, json, tempfile
import imageio.v3 as iio
import numpy as np
import pandas as pd
from bioio import BioImage
from bioio_tifffile import Reader as bioio_tifffile_reader
from bioio_ome_tiff import Reader as bioio_ome_tiff_reader
from bioio_czi import Reader as bioio_czi_reader
from bioio_imageio import Reader as bioio_imageio_reader
from bioio_lif import Reader as bioio_lif_reader
from bioio_nd2 import Reader as bioio_nd2_reader
from liffile import LifFile
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.utils.misc import md2dataframe

def downsample_for_png(arr: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    """Downsample a (Y,X,C) uint8 array so neither Y nor X exceeds max_dim."""
    from skimage.transform import resize

    y, x = arr.shape[:2]
    if max(y, x) <= max_dim:
        return arr

    scale = max_dim / max(y, x)
    new_shape = (max(1, int(y * scale)), max(1, int(x * scale)))
    print(f'Downsampling PNG from {(y, x)} to {new_shape} to keep longest side under {max_dim}px')

    out_shape = new_shape if arr.ndim == 2 else (*new_shape, arr.shape[2])
    resized = resize(arr, out_shape, anti_aliasing=True, preserve_range=True)
    return resized.astype(np.uint8)

def _normalise_to_uint8(arr, global_max=None, global_min=None):
    arr = arr.astype(np.float32)
    max_arr = global_max if global_max is not None else arr.max()
    min_arr = global_min if global_min is not None else arr.min()
    if max_arr == min_arr:
        return np.zeros_like(arr, dtype=np.uint8) # uniform (black)
    return ((arr - min_arr) / (max_arr - min_arr) * 255.0).astype(np.uint8)

READERS = {
        'imageio': {'reader': bioio_imageio_reader, 'ext': ['.png', '.jpg', '.jpeg', '.webp', '.gif']},
        'czi': {'reader': bioio_czi_reader, 'ext': ['.czi']},
        'lif':  {'reader': bioio_lif_reader, 'ext': ['.lif']},
        'nd2':  {'reader': bioio_nd2_reader, 'ext': ['.nd2']},
        'ome_tiff':  {'reader': bioio_ome_tiff_reader, 'ext': ['.ome.tif', '.ome.tiff']}, # Note must be checked BEFORE tiff
        'tifffile':  {'reader': bioio_tifffile_reader, 'ext': ['.tif', '.tiff']},
}

SUPPORTED_EXT = [ext for reader in READERS for ext in READERS[reader]['ext']]

def _resolve_path(path_or_bytes: str | bytes, filename: str | None = None):
    """Return (path, is_temporary) for an existing file path or 
    tempoary file for bytes input (keep track of is_temporary so can delete later)"""
    if isinstance(path_or_bytes, bytes):
        assert filename is not None, 'filename required when passing bytes'
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(path_or_bytes)
            tmp.flush()
        return tmp.name, True
    return path_or_bytes, False

def bioio_loader(path_or_bytes: str | bytes, filename: str | None = None):
    path, is_tmp = _resolve_path(path_or_bytes, filename)
    return _bioio_loader(path, filename)

def _bioio_loader(path: str, original_name: str | None = None):
    freader = None
    reader_name = None
    fname = original_name if original_name is not None else os.path.basename(path)
    for _reader_name, _reader_val in READERS.items():
        if any(path.endswith(ext) for ext in _reader_val['ext']):
            reader_name = _reader_name
            freader = _reader_val['reader']
            break
        #for ext in reader_val['ext']:
        #    if path.endswith(ext):
        #        fext = ext
        #        freader = reader_val['reader']
        #        break
    if freader is None:
        raise TypeError(f'Unsupported file extension for {fname}')
    # reader_name should be set
    # Load image using identified reader
    img = BioImage(path, reader=freader)
    # dimensions object https://bioio-devs.github.io/bioio/generated/bioio.Dimensions.html#bioio.Dimensions
    dims_str = ','.join([d for d in img.dims._order])
    print(f'Loaded image from {fname} using {reader_name} reader.')
    print(f'Image dimensions: ({dims_str}) = {img.shape}')
    return img

def inspect_dims(path_or_bytes: str | bytes, fname: str | None = None) -> tuple[str, tuple]:
    img = bioio_loader(path_or_bytes, fname)
    return img.dims._order, img.shape

def _standard_metadata_df(img: BioImage) -> pd.DataFrame:
    """Cross-format curated metadata (pixel sizes, objective, etc.), free since img is already loaded."""
    sm = img.standard_metadata.to_dict()
    return pd.DataFrame(list(sm.items()), columns=['Parameter', 'Value'])

def _read_czi_metadata(path: str) -> pd.DataFrame:
    """Full CZI metadata dump via czitools -> pandas DataFrame."""
    mdata = CziMetadata(path)
    return md2dataframe(mdata, reduced_params=False)

def _flatten_dict(d: dict, parent_key: str = '') -> dict:
    """Flatten a nested dict into dotted-key: value pairs."""
    items = {}
    for k, v in d.items():
        key = f'{parent_key}.{k}' if parent_key else str(k)
        if isinstance(v, dict):
            # recurse to handle nested dictionaries
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items

def _read_lif_metadata(path: str):
    """LIF metadata (liffile) -> pandas DataFrame"""
    with LifFile(path) as lif:
        if not lif.images:
            return pd.DataFrame(columns=['Parameter', 'Value'])
        image = lif.images[0] # N.B. Take first image metadata ONLY (limitation)
        flat = _flatten_dict(dict(image.attrs))
        flat['Name'] = image.name
        flat['Sizes'] = image.sizes
    return pd.DataFrame(list(flat.items()), columns=['Parameter', 'Value'])

def inspect_metadata(path_or_bytes: str | bytes, fname: str | None = None, img: BioImage | None = None):
    """
    Extra image metadata to a pandas DataFrame. Load from path or image bytes (requires a file name)
    or use a BioImage object that was already loaded
    """
    if not isinstance(path_or_bytes, str):
        assert fname is not None, 'fname required when passing bytes'
    else:
        fname = path_or_bytes
    path, is_tmp = _resolve_path(path_or_bytes, fname)
    if img is None:
        # Image has not been loaded yet, so load it
        img = _bioio_loader(path, fname)
    frames = [_standard_metadata_df(img)]
    if fname.endswith('.czi'):
        frames.append(_read_czi_metadata(path))
    elif fname.endswith('.lif'):
        frames.append(_read_lif_metadata(path))
    return pd.concat(frames, ignore_index=True)

def inspect_image(path_or_bytes: str | bytes, fname: str | None = None) -> dict:
    """
    Inspect a microscopy image: filetype, dims/shape, and metadata
    """
    if not isinstance(path_or_bytes, str):
        assert fname is not None, 'fname required when passing bytes'
    else:
        fname = path_or_bytes
    path, is_tmp = _resolve_path(path_or_bytes, fname)
    img = _bioio_loader(path, fname)
    filetype = os.path.splitext(fname)[1].lstrip('.').lower()
    return {
        'filetype': filetype,
        'dims': img.dims._order,
        'shape': img.shape,
        'metadata': inspect_metadata(path, fname, img=img),  # reuses img, no second load
    }

def prep_pipeline_image(image: np.ndarray):
    """
    Convert original image data into uint8 or uint16 format to pass to analysis pipeline:
        - Takes into account the range and precision of the input array. 
        - Normalises floating point arrays with over 20% negative values to avoid clipping during conversion.
    """
    from skimage.util import img_as_ubyte, img_as_uint
    from skimage.exposure import rescale_intensity

    type = image.dtype
    
    #find out of floating point or integer
    if np.issubdtype(type, np.integer) == True:
        if np.issubdtype(type,np.uint8) == True:
            # do nothing, already uint8
            pass
        elif np.issubdtype(type,np.uint16) == True:
            if len(np.unique(image)) <= 256 and np.ptp(image) <= 256:
                # convert to uint8 if the data range and precision are both within uint8 range (sparse uint16)
                # might cause contrast scaling that is unideal, but unlikely
                image = img_as_ubyte(image)
            else:
                # do nothing, already full uint16
                pass
        else:
            # convert other int data types (eg signed int) to uint8/16 based on data precision/range
            if len(np.unique(image)) <= 256 and np.ptp(image) <= 256:
                image = img_as_ubyte(image)
            else:
                image = img_as_uint(image)
    
    elif np.issubdtype(image.dtype,np.bool) == True:
        # convert boolean masks to uint with [0,255] range
        image = img_as_ubyte(image)

    elif np.issubdtype(image.dtype, np.floating) == True:
        if len(np.unique(image)) <= 256 and np.ptp(image) <= 1:
            # if we accidentally get a FP representation of data that is actually uint8 precision
            # doesn't account for negative pixels or clipping
            image = img_as_ubyte(image)
        else:
            if (image<0).sum()/image.size >= 0.2:
                # suggests that the FP image has significant pixels with negative values
                # which will be clipped by default during conversion
                # therefore we normalise to between 0 and 1 prior to conversion to uint16
                image = rescale_intensity(image, out_range=(0, 1))
                image = img_as_uint(image)
                
            else:
                image = img_as_uint(image)
                
    else:
        try:
            image = img_as_uint(image)
        except:
            raise TypeError("Can't convert input image data type.")

    return image

def dataframe_to_dict(df: pd.DataFrame, paramcol: str = 'Parameter', keycol: str = 'Value') -> dict:
    """Flatten a Parameter/Value metadata DataFrame into a plain dict."""
    return dict(zip(df[paramcol], df[keycol]))

def metadata_to_json(df: pd.DataFrame, **kwargs) -> str:
    """Serialise a metadata DataFrame to a JSON string, e.g. for inclusion in an LLM prompt."""
    return json.dumps(dataframe_to_dict(df, **kwargs), default=str, indent=2)

def load_image(
        path_or_bytes: str | bytes,
        fname: str | None = None,
        dims_override: str = None,
        t: int = 0,
        z: int | None = None,
        z_mode: str | None = None,
        channel: int | None = None,
        per_channel_normalise: bool = True,
        max_png_dim: int = 2048,
        ) -> tuple[bytes, np.ndarray]:
    """
    Load a microscopy image, applying the requested t/z/channel selection,
    and return both:
      - png_bytes: a normalised uint8 PNG suitable for LLM/display use
      - original_image_data: the same t/z-selected slice, at native pixel
        values (dtype unchanged from the source file). If a single channel
        was explicitly selected, this is a 2D+1 (Y,X,C) array; otherwise it
        includes ALL channels (not just the first three used for the PNG
        composite), shape (Y,X,C).

    """
    assert z is None or z_mode is None, 'Only one of parameters z, z_max should be set'
    mode_to_func = {'max': np.max, 'min': np.min, 'mean': np.mean}
    if z_mode is not None:
        assert z_mode in mode_to_func, "z_mode must be 'max', 'min', 'mean' or None"

    img = bioio_loader(path_or_bytes, fname)
    dims_str = img.dims._order

    # Check we can recognise 2d data
    for d in ['X','Y']:
        assert d in dims_str, f'No {d}-coordinate data found. Data may be missing or mislabelled.'

    if dims_override is not None:
        assert set(dims_override) == set(dims_str), \
            f'dims_override {dims_override!r} must contain same axes as detected: {dims_str}'
        print(f'Overriding detected dim order {dims_str} with {dims_override}')
        dims_str = dims_override
    data = np.asarray(img.get_image_data(dims_str))

    # canonical Order TCZYX or TCZYXS
    if  data.shape[0] > 1:
        print(f'Taking timepoint T={t}')
    data = data[t]
    # ~~~ CZYX ~~~
    if 'S' in dims_str:
        # CZYXS; Discard C (take first element)
        # 'Samples' dimension contains image value e.g. RGB at each coordinate
        if data.shape[0] > 1:
            print('WARNING: Taking data across Sample dimension but Channel dimension is non-trivial; only data from first channel will be used')
        data = data[0]
    else:
        # Reorder as ZYXC
        data = np.moveaxis(data, 0, -1)
    # ~~~ ZYXC ~~~ (Not 'C' may be 'S'; have same role here)
    # Z-projection
    if z_mode is not None:
        if data.shape[0] > 1:
            print(f'Taking {z_mode} value along Z axis')
        data = mode_to_func[z_mode](data, axis=0) # CYX
        z_idx = None
    else:
        z_idx = z if z is not None else data.shape[0] // 2
        if data.shape[0] > 1:
            print(f'Taking Z={z_idx} slice')
        data = data[z_idx]
    # ~~~ YXC ~~~
    # Channel projection
    num_channels = data.shape[-1]
    print(f'Reduced to 2D image data of shape (Y,X) = {data.shape[:2]} with {num_channels} channel(s) per pixel')

    # ~~~ Derive the pipeline-facing array: native dtype, reflects the
    # same t/z/channel selection, but keeps ALL channels in composite mode 
    if channel is not None:
        original_image_data = data[:, :, channel]
    else:
        original_image_data = data
        # N.B. for RGB images could consider swapping to BGR for cv2 
        # (see encode/decode_png in image_tools.py)

    # ~~~ Convert image datatype for pipeline
    prepared_image_data = prep_pipeline_image(original_image_data)

    # ~~~ Continue with normalisation for the LLM PNG
    _png_channel = channel
    # Encode to PNG bytes
    alpha = None
    if num_channels == 1:
        _png_channel = 0
    elif num_channels > 3 and _png_channel is None:
        print('Loading channels 0-2 as RGB; specify channel index to instead select a single channel')
        if num_channels == 4:
            print('Assuming 4th channel as alpha and compositing over white background')
            alpha = data[:, :, 3:4] / np.float64(255.0)
        data = data[:, :, :3]

    if _png_channel is not None:
        print(f'Selecting channel {_png_channel} normalised to uint8 (0,255)')
        norm_data = _normalise_to_uint8(data[:, :, _png_channel])
    else:
        global_max, global_min = None, None
        if per_channel_normalise:
            print('Normalising channels individually to uint8 (0,255)')
        else:
            print('Normalising channels collectively to uint8 (0,255)')
            global_max, global_min = np.max(data), np.min(data)
        r = _normalise_to_uint8(data[:, :, 0], global_max, global_min)
        g = _normalise_to_uint8(data[:, :, 1], global_max, global_min)
        if num_channels == 2:
            print('Assuming channels 0-1 as R+G image')
            b = np.ones_like(r)
        else:
            b = _normalise_to_uint8(data[:, :, 2], global_max, global_min)
        norm_data = np.stack([r, g, b], axis=-1)
        if alpha is not None:
            norm_data = (norm_data * alpha + 255 * (1 - alpha)).astype(np.uint8)
    #print(norm_data)
    norm_data = downsample_for_png(norm_data, max_dim=max_png_dim) # limit max resolution (and so roughly max filesize)
    buf = io.BytesIO()
    #print(f"norm_data dtype: {norm_data.dtype}, min: {norm_data.min()}, max: {norm_data.max()}, shape: {norm_data.shape}")
    iio.imwrite(buf, norm_data, extension='.png')
    png_bytes = buf.getvalue()

    return png_bytes, prepared_image_data
