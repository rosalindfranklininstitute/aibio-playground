import marimo

app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import sys
    sys.path.insert(0, '/app')
    from agent.image_loader import (
        inspect_image,
        inspect_metadata,
        load_to_png,
        build_image_message,
        dataframe_to_dict,
    )
    return build_image_message, inspect_image, inspect_metadata, load_to_png, mo, dataframe_to_dict


@app.cell
def __(mo):
    mo.md("# image_loader test notebook")
    return


@app.cell
def __(mo):
    file_upload = mo.ui.file(filetypes=[".czi", ".lif", ".tif", ".tiff", ".png", ".jpg"], label="Upload a microscopy image")
    file_upload
    return (file_upload,)


@app.cell
def __(file_upload, mo):
    mo.stop(not file_upload.value, mo.md("_Waiting for a file upload..._"))
    upload = file_upload.value[0]
    fname = upload.name
    file_bytes = upload.contents
    mo.md(f"Uploaded **{fname}** ({len(file_bytes):,} bytes)")
    return fname, file_bytes, upload


@app.cell
def __(fname, file_bytes, inspect_image, mo):
    result = inspect_image(file_bytes, fname)
    mo.md(f"""
    **Filetype:** `{result['filetype']}`

    **Dims:** `{result['dims']}`

    **Shape:** `{result['shape']}`
    """)
    return (result,)


@app.cell
def __(mo, result):
    mo.ui.table(result["metadata"], selection=None, label="Metadata (scrollable)")
    return


@app.cell
def __(fname, file_bytes, load_to_png, mo):
    png_bytes = load_to_png(file_bytes, fname)
    mo.image(png_bytes)
    return (png_bytes,)


@app.cell
def __(mo):
    mo.md("## LLM message preview")
    return


@app.cell
def __(build_image_message, file_upload, mo, result, dataframe_to_dict):
    metadata_dict = dataframe_to_dict(result["metadata"])
    message = build_image_message(
        "What do you observe in this image?", file_upload, metadata=metadata_dict
    )
    mo.md(f"""
    Metadata-prefixed text sent to the LLM:

    ```
    {message["content"][1]["text"] if isinstance(message["content"], list) else message["content"]}
    ```
    """)
    return message, metadata_dict


if __name__ == "__main__":
    app.run()
