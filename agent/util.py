import io, sys, os, base64, tempfile
import imageio.v3 as iio
import numpy as np
from bioio import BioImage
from bioio_tifffile import Reader as bioio_tifffile_reader
from bioio_ome_tiff import Reader as bioio_ome_tiff_reader
from bioio_czi import Reader as bioio_czi_reader
from bioio_imageio import Reader as bioio_imageio_reader
from bioio_lif import Reader as bioio_lif_reader
from bioio_nd2 import Reader as bioio_nd2_reader

def build_image_message(text, file_upload):
    """
    Encode an image uploaded to marimo and package into a multipart
    message dict (OpenAI spec) along with a user text prompt.

    Parameters
    ----------
    text : str
        The user's text message.
    file_upload : mo.ui.file
        Marimo file upload element.

    Returns
    -------
    dict
        Message dict with role and content that can be sent to Ollama
    """
    if not file_upload.value:
        return {"role": "user", "content": text}

    image_data = base64.b64encode(
        file_upload.value[0].contents
    ).decode('utf-8')

    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            },
            {
                "type": "text",
                "text": text
            }
        ]
    }

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

def bioio_loader(path_or_bytes: str | bytes, filename: str | None = None) -> BioImage:
    if isinstance(path_or_bytes, bytes):
        assert filename is not None, 'filename required when passing bytes'
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(path_or_bytes)
            tmp.flush()
            return _bioio_loader(tmp.name, filename)
    return _bioio_loader(path_or_bytes)

def _bioio_loader(path: str, original_name: str | None = None) -> BioImage:
    freader = None
    fname = original_name if original_name is not None else os.path.basename(path)
    for reader_name, reader_val in READERS.items():
        for ext in reader_val['ext']:
            if path.endswith(ext):
                fext = ext
                freader = reader_val['reader']
                break
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


def load_to_png(
        path_or_bytes: str | bytes,
        fname: str | None = None,
        dims_override: str = None,
        t: int = 0,
        z: int | None = None,
        z_mode: str | None = None,
        channel: int | None = None,
        per_channel_normalise: bool = True,
        ) -> bytes:
    """
    Load a microscopy image from path to PNG suitable for LLM use

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
        # need to rotate (swap X and Y)?
        # data = np.swapaxes(data, 1, 2)
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

    alpha = None
    if num_channels == 1:
        channel = 0
    elif num_channels > 3 and channel is None:
        print('Loading channels 0-2 as RGB; specify channel index to instead select a single channel')
        if num_channels == 4:
            print('Assuming 4th channel as alpha and compositing over white background')
            alpha = data[:,:,3:4] / np.float64(255.0)
        data = data[:, :, :3]
    if channel is not None:
        # Single channel selection
        print(f'Selecting channel {channel} normalised to uint8 (0,255)')
        norm_data = _normalise_to_uint8(data[:,:,channel])
    else:
        global_max, global_min = None, None
        if per_channel_normalise:
            print('Normalising channels individually to uint8 (0,255)')
        else:
            print('Normalising channels collectively to uint8 (0,255)')
            global_max, global_min = np.max(data), np.min(data)
        r = _normalise_to_uint8(data[:,:,0], global_max, global_min)
        g = _normalise_to_uint8(data[:,:,1], global_max, global_min)
        if num_channels == 2:
            print('Assuming channels 0-1 as R+G image')
            b = np.ones_like(r)
        else:
            b = _normalise_to_uint8(data[:,:,2], global_max, global_min)
        norm_data = np.stack([r, g, b], axis=-1)
        if alpha is not None:
            norm_data = (norm_data * alpha + 255 * (1 - alpha)).astype(np.uint8)
    # Encode to PNG bytes
    #print(norm_data)
    buf = io.BytesIO()
    #print(f"norm_data dtype: {norm_data.dtype}, min: {norm_data.min()}, max: {norm_data.max()}, shape: {norm_data.shape}")
    iio.imwrite(buf, norm_data, extension='.png')
    png_bytes = buf.getvalue()

    metadata = {} # Todo e.g. and dimensional data from initial image

    return png_bytes


