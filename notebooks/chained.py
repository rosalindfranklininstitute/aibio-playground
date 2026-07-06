import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import inspect
    import base64
    from wigglystuff import SortableList, CellTour
    sys.path.insert(0, '/app')
    from agent.loader import load_catalogue
    from agent.pipeline_builder import step, ConversationState, get_client, run_pipeline
    from agent.image_tools import decode_image, encode_png

    return (
        ConversationState,
        SortableList,
        base64,
        decode_image,
        encode_png,
        get_client,
        inspect,
        load_catalogue,
        mo,
        run_pipeline,
        step,
    )


@app.cell
def _():
    tour = mo.ui.anywidget(
        CellTour(
            steps=[
                {
                    "cell_name": "upload_image_step",
                    "title": "1. Upload an image",
                    "description": "Upload your image here. Currently only RGB images are supported.",
                },
                {
                    "cell_name": "ask_agent_step",
                    "title": "2. Ask the agent",
                    "description": "Tell the agent what you want to achieve. It will ask clarifying questions before proposing a pipeline.",
                },
                {
                    "cell_name": "pipeline_sortablelist",
                    "title": "3. View the pipeline",
                    "description": "The proposed pipeline will appear here. You can reorder or remove functions if needed.",
                },
                {
                    "cell_name": "run_pipeline_step",
                    "title": "4. Run the pipeline",
                    "description": "Press run to apply the pipeline to your image data. The result appears in a box below."
                },
                {
                    "cell_name": "available_functions_step",
                    "title": "5. extra functions",
                    "description": "If the agent missed something, tick functions here and add them to the pipeline manually.",
                },
            ]
        )
    )
    tour
    return (tour,)


@app.cell
def _(load_catalogue):
    catalogue = load_catalogue()
    return (catalogue,)


@app.cell
def _(get_client):
    client = get_client()
    return (client,)


@app.cell
def _(mo):
    mo.md("## Upload Image")
    return


@app.cell
def _(ConversationState, mo):
    get_image_sent, set_image_sent = mo.state(False)
    get_conv_state, set_conv_state = mo.state(ConversationState())
    return get_conv_state, get_image_sent, set_conv_state, set_image_sent


@app.cell
def _(ConversationState, set_conv_state, set_image_sent):
    def new_converation(_):
        set_image_sent(False)
        set_conv_state(ConversationState())
    return (new_converation,)


@app.cell
def _(new_converation, mo):
    file_upload = mo.ui.file(
        label="Upload an image",
        on_change=new_converation,
    )
    return (file_upload,)


@app.cell
def upload_image_step(file_upload):
    file_upload
    return


@app.cell
def _(mo):
    mo.md("## Ask the Agent")
    return


@app.cell
def _(mo):
    get_pipeline, set_pipeline = mo.state([])
    return get_pipeline, set_pipeline


@app.cell
def _(mo):
    # Maps function name -> args dict. Kept in sync with get_pipeline.
    # Is this the best way to manage state?
    get_pipeline_args, set_pipeline_args = mo.state({})
    return get_pipeline_args, set_pipeline_args


@app.cell
def _(mo):
    get_reasoning, set_reasoning = mo.state(None)
    return get_reasoning, set_reasoning


@app.cell
def _(
    base64,
    catalogue,
    client,
    file_upload,
    get_conv_state,
    get_image_sent,
    mo,
    set_conv_state,
    set_image_sent,
    set_pipeline,
    set_pipeline_args,
    set_reasoning,
    step,
):
    def chat_agent(messages, config):
        image_b64 = None
        # TODO: include import-to-png logic to handle scientific images etc.
        if file_upload.value and not get_image_sent():
            image_b64 = base64.b64encode(file_upload.value[0].contents).decode('utf-8')
            set_image_sent(True)

        user_message = messages[-1].content
        result = step(get_conv_state(), user_message, catalogue, client, image_b64=image_b64)
        set_conv_state(result['state'])
        set_reasoning(result['reasoning'])

        if result['pipeline'] is not None:
            set_pipeline([s['name'] for s in result['pipeline']])
            set_pipeline_args({s['name']: s['args'] for s in result['pipeline']})

        return result['message']

    chat = mo.ui.chat(
        chat_agent,
        max_height=500,
        disabled=not file_upload.value,
    )
    return (chat,)


@app.cell
def ask_agent_step(chat):
    chat
    return


@app.cell
def _(get_reasoning, mo):
    _reasoning = get_reasoning()
    if _reasoning:
        thinking = mo.accordion({"Model thinking": mo.md(_reasoning)})
    else:
        thinking = mo.md("")
    return (thinking,)


@app.cell
def _(thinking):
    thinking
    return


@app.cell
def _(mo):
    mo.md("## Pipeline")
    return


@app.cell
def pipeline_sortablelist(SortableList, get_pipeline, mo):
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
def _(catalogue, get_pipeline_args, pipeline_widget, set_pipeline_args):
    # Sync pipeline_args with SortableList's state
    # drop args for removed functions, fill defaults for names added via table
    _names = pipeline_widget.value.get("value", [])
    _current_args = get_pipeline_args()
    _synced_args = {
        n: _current_args.get(n, catalogue[n]['defaults'])
        for n in _names
        if n in catalogue
    }
    if _synced_args != _current_args:
        set_pipeline_args(_synced_args)
    return


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
    mo.md("## Run")
    return


@app.cell
def _(mo, pipeline_widget):
    _names = pipeline_widget.value.get("value", [])
    # https://www.alt-codes.net/arrow_alt_codes.php
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
def run_pipeline_step(run_button):
    run_button
    return


@app.cell
def _(
    catalogue,
    decode_image,
    encode_png,
    file_upload,
    get_pipeline_args,
    mo,
    pipeline_widget,
    run_button,
    run_pipeline,
):
    mo.stop(not run_button.value, mo.md("Press Run to execute."))
    mo.stop(not file_upload.value, mo.md("Upload an image first."))

    _names = pipeline_widget.value.get("value", [])
    mo.stop(not _names, mo.md("No functions in pipeline."))

    _args_map = get_pipeline_args()
    _pipeline = [
        {"name": n, "args": _args_map.get(n, catalogue[n]['defaults'])}
        for n in _names
        if n in catalogue
    ]

    _image = decode_image(file_upload.value[0].contents)
    _output = run_pipeline(_image, _pipeline, catalogue)

    mo.image_compare(
        encode_png(_image),
        encode_png(_output),
        width=_image.shape[1],
    )
    return


@app.cell
def _(mo):
    mo.md("## Available Functions")
    return


@app.cell
def available_functions_step(catalogue, mo):
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
    add_button = mo.ui.run_button(label="Add functions selected in table to pipeline")
    add_button
    return (add_button,)


@app.cell
def _(add_button, catalogue_table, get_pipeline, set_pipeline):
    if add_button.value and catalogue_table.value:
        selected_names = [row['name'] for row in catalogue_table.value]
        current = get_pipeline()
        # TODO: ability to have functions repeated with separate arguments
        # currently just prevent duplications as they would have the same argument
        # e.g. assign id to each function
        # to_add = [n for n in selected_names if n not in current] # comment -> duplicates allowed (same arguments)
        set_pipeline(current + to_add)
    return


if __name__ == "__main__":
    app.run()
