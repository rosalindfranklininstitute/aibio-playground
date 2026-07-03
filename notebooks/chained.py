import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import inspect
    import numpy as np
    import cv2
    from functools import partial, reduce
    from wigglystuff import SortableList
    sys.path.insert(0, '/app')
    from agent.loader import load_catalogue
    from agent.tool_selector import select_pipeline
    from agent.util import build_image_message

    return (
        SortableList,
        build_image_message,
        cv2,
        inspect,
        load_catalogue,
        mo,
        np,
        partial,
        reduce,
        select_pipeline,
    )


@app.cell
def _(load_catalogue):
    catalogue = load_catalogue()
    return (catalogue,)


@app.cell
def _(mo):
    mo.md("""
    ## Upload Image
    """)
    return


@app.cell
def _(mo):
    get_image_sent, set_image_sent = mo.state(False)
    return get_image_sent, set_image_sent


@app.cell
def _(mo, set_image_sent):
    file_upload = mo.ui.file(
        label="Upload an image",
        on_change=lambda _: set_image_sent(False)
    )
    return (file_upload,)


@app.cell
def _(file_upload):
    file_upload
    return


@app.cell
def _(mo):
    mo.md("""
    ## Available Functions
    """)
    return


@app.cell
def _(catalogue, mo):
    catalogue_table = mo.ui.table(
        data=[
            {
                "name": name,
                "description": entry['metadata']['description'][:100] + "..."
            }
            for name, entry in catalogue.items()
        ],
        selection="multi",
        label="Available Functions",
    )
    catalogue_table
    return (catalogue_table,)


@app.cell
def _(mo):
    mo.md("""
    ## Ask the Agent
    """)
    return


@app.cell
def _(mo):
    get_pipeline, set_pipeline = mo.state([])
    return get_pipeline, set_pipeline


@app.cell
def _(mo):
    get_result, set_result = mo.state(None)
    return get_result, set_result


@app.cell
def _(
    build_image_message,
    catalogue,
    file_upload,
    get_image_sent,
    mo,
    select_pipeline,
    set_image_sent,
    set_pipeline,
    set_result,
):
    def chat_agent(messages, config):
        msgs = list(messages)
        if file_upload.value and not get_image_sent():
            msgs[-1] = build_image_message(messages[-1].content, file_upload)
            set_image_sent(True)
        result = select_pipeline(msgs, catalogue)
        if result['selected']:
            set_result(result)
            set_pipeline([step['name'] for step in result['pipeline']])
        return result['message']

    chat = mo.ui.chat(chat_agent, max_height=500)
    return (chat,)


@app.cell
def _(chat):
    chat
    return


@app.cell
def _(chat, get_result, mo):
    mo.stop(not chat.value, mo.md("No messages yet."))
    result = get_result()
    mo.stop(result is None, mo.md("No pipeline selected yet."))
    if result.get('reasoning'):
        thinking = mo.accordion({"Model thinking": mo.md(result['reasoning'])})
    else:
        thinking = mo.md("")
    return (thinking,)


@app.cell
def _(thinking):
    thinking
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pipeline
    """)
    return

@app.cell
def _(mo):
    add_button = mo.ui.run_button(label="Add functions selected in table to pipeline")
    add_button
    return (add_button,)

@app.cell
def _(add_button, catalogue_table, get_pipeline, set_pipeline):
    if add_button.value and catalogue_table.value:
        selected_names = [row['name'] for row in catalogue_table.value]
        current = get_pipeline()
        # Avoid duplicates
        to_add = [n for n in selected_names if n not in current]
        set_pipeline(current + to_add)
    return


@app.cell
def _(SortableList, get_pipeline, mo):
    pipeline_widget = mo.ui.anywidget(
        SortableList(
            get_pipeline(),
            editable=False,
            addable=False,
            removable=True,
            label="Pipeline (drag to reorder, click X to remove)",
        )
    )
    pipeline_widget
    return (pipeline_widget,)


@app.cell
def _(catalogue, inspect, mo, pipeline_widget):
    _names = pipeline_widget.value.get("value", [])
    if _names:
        source_accordion = mo.accordion({
            name: mo.ui.code_editor(
                value=inspect.getsource(catalogue[name]['function']),
                language="python",
            )
            for name in _names
            if name in catalogue
        })
    else:
        source_accordion = mo.md("_No functions in pipeline yet._")
    source_accordion
    return


@app.cell
def _(mo):
    mo.md("""
    ## Run
    """)
    return


@app.cell
def _(mo, pipeline_widget):
    _names = pipeline_widget.value.get("value", [])
    mo.md(
        f"Running pipeline: `{'` → `'.join(_names)}`"
        if _names else "_No functions selected._"
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run pipeline")
    return (run_button,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(
    catalogue,
    cv2,
    file_upload,
    mo,
    np,
    partial,
    pipeline_widget,
    reduce,
    run_button,
):
    mo.stop(not run_button.value, mo.md("Press Run to execute."))
    mo.stop(not file_upload.value, mo.md("Upload an image first."))

    _names = pipeline_widget.value.get("value", [])
    mo.stop(not _names, mo.md("No functions in pipeline."))

    # Decode uploaded image
    _raw = np.frombuffer(file_upload.value[0].contents, np.uint8)
    _image = cv2.imdecode(_raw, cv2.IMREAD_COLOR)

    # Build partials from catalogue defaults
    _sequence = [
        partial(catalogue[name]['function'], **catalogue[name]['defaults'])
        for name in _names
        if name in catalogue # safegaurd 
    ]

    # Reduce: thread image through pipeline
    _output = reduce(lambda im, fn: fn(im), _sequence, _image)

    mo.image_compare(
        cv2.imencode(".png", _image)[1].tobytes(),
        cv2.imencode(".png", _output)[1].tobytes(),
        width=_image.shape[1],
    )
    return


if __name__ == "__main__":
    app.run()
