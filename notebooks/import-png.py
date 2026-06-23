import marimo
__generated_with = "0.23.1"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import sys
    sys.path.insert(0, '/app')
    from agent.util import load_to_png, inspect_dims
    return inspect_dims, load_to_png, mo, sys

@app.cell
def _(mo):
    from agent.util import SUPPORTED_EXT
    upload = mo.ui.file(
        filetypes=SUPPORTED_EXT,
        kind="button",
        label="Upload a microscopy image",
    )
    upload
    return (upload,)

@app.cell
def _(inspect_dims, mo, upload):
    mo.stop(not upload.value, None)
    detected_dims, detected_shape = inspect_dims(upload.contents(0), upload.name(0))
    col_widths = [len(str(s)) for s in detected_shape]
    header = '  '.join(d.ljust(w) for d, w in zip(detected_dims, col_widths))
    values = '  '.join(str(s).ljust(w) for s, w in zip(detected_shape, col_widths))
    mo.md(f"**Detected dimensions:**\n```\n{header}\n{values}\n```")
    return detected_dims, detected_shape

@app.cell
def _(detected_dims, mo):
    dims_input = mo.ui.text(
        value=detected_dims,
        label='Override axes order:',
    )
    return (dims_input,)

@app.cell
def _(dims_input, detected_dims):
    dims_override = dims_input.value if dims_input.value != detected_dims else None
    return (dims_override,)

@app.cell
def _(mo, dims_override, detected_shape):
    # Can't limit max t-index (similarly z-index) in case user specifies dims_override
    if dims_override is None:
        tstop = detected_shape[0]-1 # Assumes canonical order
    else:
        tstop = None
    t_input = mo.ui.number(start=0, step=1, stop=tstop, value=0, label="t-index")
    return (t_input,)

@app.cell
def _(mo):
    z_mode_dropdown = mo.ui.dropdown(
        options={"(none)": None, "max": "max", "min": "min", "mean": "mean"},
        value="(none)",
        label="Operation to apply along z-axis for stacked data",
    )
    return (z_mode_dropdown,)

@app.cell
def _(mo, z_mode_dropdown, dims_override):
    if dims_override is None:
        zstop = detected_shape[2]-1
    else:
        zstop = None
    if z_mode_dropdown.value is not None:
        z_input = mo.md("_z disabled: operation applied along Z axis_")
    else:
        z_input = mo.ui.number(
            start=0,
            step=1,
            stop=zstop,
            value=None,
            label="z-stack index",
        )
    return (z_input,)

@app.cell
def _(z_input, z_mode_dropdown):
    z_value = None if z_mode_dropdown.value is not None else z_input.value
    return (z_value,)

@app.cell
def _(mo):
    channel_none = mo.ui.checkbox(value=True, label="Composite (selects up to first three channels)")
    return (channel_none,)

@app.cell
def _(channel_none, mo):
    channel_input = mo.ui.number(
        start=0, step=1, value=0, label="Selected channel index"
    ) if not channel_none.value else None
    per_channel_norm = mo.ui.checkbox(
        value=True, label="Normalise channels individually"
    )
    return channel_input, per_channel_norm

@app.cell
def _(channel_input, channel_none, dims_input, mo, per_channel_norm, t_input, z_input, z_mode_dropdown):
    mo.vstack([
        mo.md("### Parameters"),
        dims_input,
        t_input,
        z_mode_dropdown,
        z_input,
        channel_none,
        channel_input if channel_input is not None else mo.md("_Composite_"),
        per_channel_norm,
    ])
    return

@app.cell
def _(mo, upload):
    load_button = mo.ui.run_button(
        label='Load image',
        disabled=not upload.value,
    )
    load_button
    return (load_button,)

@app.cell
def _(channel_input, channel_none, dims_override, load_button, load_to_png, mo, per_channel_norm, t_input, upload, z_mode_dropdown, z_value):
    mo.stop(not load_button.value, mo.md("_Upload an image and click Load to continue._"))
    with mo.capture_stdout() as stdout_buf:
        png_bytes = load_to_png(
            upload.contents(0),
            fname=upload.name(0),
            dims_override=dims_override,
            t=int(t_input.value),
            z=int(z_value) if z_value is not None else None,
            z_mode=z_mode_dropdown.value,
            channel=int(channel_input.value) if not channel_none.value else None,
            per_channel_normalise=per_channel_norm.value,
        )
    log_text = stdout_buf.getvalue()
    return log_text, png_bytes

@app.cell
def _(log_text, mo, png_bytes):
    mo.vstack([
        mo.md("### Load log"),
        mo.md(f"```\n{log_text}\n```"),
        mo.md("### Result"),
        mo.image(png_bytes),
    ])
    return

if __name__ == "__main__":
    app.run()
