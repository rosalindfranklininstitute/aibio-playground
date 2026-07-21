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
    from agent.catalogue_loader import load_catalogue
    from agent.pipeline_builder import step, ConversationState, get_client, run_pipeline, generate_step_id, step_id_to_name
    from agent.image_tools import decode_image, encode_png

    return (
        CellTour,
        ConversationState,
        SortableList,
        base64,
        decode_image,
        encode_png,
        generate_step_id,
        get_client,
        inspect,
        load_catalogue,
        mo,
        run_pipeline,
        step,
        step_id_to_name,
    )


@app.cell
def _(CellTour, mo):
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
                    "title": "5. Add extra functions",
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
    mo.md("## Image Loader")
    return


@app.cell
def _(ConversationState, mo):
    get_image_sent, set_image_sent = mo.state(False)
    get_conv_state, set_conv_state = mo.state(ConversationState())
    return get_conv_state, get_image_sent, set_conv_state, set_image_sent


@app.cell
def _(ConversationState, set_conv_state, set_image_sent):
    def new_conversation(_):
        set_image_sent(False)
        set_conv_state(ConversationState())
    return (new_conversation,)


@app.cell
def _(new_conversation, mo):
    file_browser = mo.ui.file_browser(
        initial_path='/app/assets',
        filetypes=None,
        selection_mode='file',
        multiple=False,
        on_change=new_conversation,
        label='Select an image file',
    )
    return (file_browser,)


@app.cell
def upload_image_step(file_browser):
    file_browser
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
    # Maps step id -> args dict. Kept in sync with get_pipeline via
    # the args_form cell (below), which seeds defaults for any step
    # id lacking an entry and writes back user edits.
    get_pipeline_args, set_pipeline_args = mo.state({})
    return get_pipeline_args, set_pipeline_args


@app.cell
def _(mo):
    # DEBUG: seed a known pipeline directly, bypassing the agent.
    # Useful for testing the pipeline/args/run cells on hardware too
    # small to run the LLM comfortably. Remove once no longer needed.
    seed_button = mo.ui.run_button(label="Seed example pipeline (debug)")
    seed_button
    return (seed_button,)


@app.cell
def _(seed_button, set_pipeline, set_pipeline_args):
    if seed_button.value:
        set_pipeline(["gaussian_blur", "simple_threshold"])
        set_pipeline_args({
            "gaussian_blur": {"kernel_size": 5, "sigma": 0},
            "simple_threshold": {"threshold": None, "invert": False},
        })
    return


@app.cell
def _(mo):
    get_reasoning, set_reasoning = mo.state(None)
    return get_reasoning, set_reasoning


@app.cell
def _(
    base64,
    catalogue,
    client,
    file_browser,
    generate_step_id,
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
        if file_browser.value and not get_image_sent():
            path = file_browser.path(index=0)
            image_b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
            set_image_sent(True)

        user_message = messages[-1].content
        result = step(get_conv_state(), user_message, catalogue, client, image_b64=image_b64)
        set_conv_state(result['state'])
        set_reasoning(result['reasoning'])

        if result['pipeline'] is not None:
            step_ids = []
            args_map = {}
            for s in result['pipeline']:
                sid = generate_step_id(s['name'], step_ids)
                step_ids.append(sid)
                args_map[sid] = s['args']
            set_pipeline(step_ids)
            set_pipeline_args(args_map)

        return result['message']

    chat = mo.ui.chat(
        chat_agent,
        max_height=500,
        disabled=not file_browser.value,
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
def _(get_pipeline, pipeline_widget, set_pipeline):
    # Detect removals (not reorders) in the SortableList and mirror
    # them into get_pipeline, so args_form correctly drops removed
    # steps. Pure reordering (same set of ids) never calls
    # set_pipeline here, so args_form's widgets stay undisturbed.
    _widget_ids = pipeline_widget.value.get("value", [])
    _tracked_ids = get_pipeline()
    if set(_widget_ids) < set(_tracked_ids):  # proper subset = something removed
        set_pipeline(_widget_ids)
    return


@app.cell
def _(mo):
    mo.md("### Edit Arguments")
    return


@app.cell
def _(mo):
    def make_widget(param_type, value):
        if param_type == "number":
            return mo.ui.number(value=value)
        elif param_type == "boolean":
            return mo.ui.checkbox(value=bool(value))
        else:
            return mo.ui.text(value=str(value))
    return (make_widget,)


@app.cell
def _(mo):
    get_args_seed, set_args_seed = mo.state({})
    return get_args_seed, set_args_seed


@app.cell
def _(catalogue, get_args_seed, get_pipeline, get_pipeline_args, set_args_seed, step_id_to_name):
    # Reseed only when the SET of step ids changes (add/remove).
    # Uses get_pipeline (not pipeline_widget) since that's stable
    # across drag-reordering — SortableList owns order, this only
    # needs to know which steps exist.
    _step_ids = get_pipeline()
    _current_seed = get_args_seed()
    _current_args = get_pipeline_args()
    _new_seed = {
        sid: _current_args.get(sid, _current_seed.get(sid, catalogue[step_id_to_name(sid)]['defaults']))
        for sid in _step_ids
        if step_id_to_name(sid) in catalogue
    }
    if set(_new_seed.keys()) != set(_current_seed.keys()):
        set_args_seed(_new_seed)
    return


@app.cell
def _(catalogue, get_args_seed, get_pipeline, make_widget, mo, step_id_to_name):
    _step_ids = get_pipeline()
    _seed = get_args_seed()

    _forms = {}
    for _step_id in _step_ids:
        _name = step_id_to_name(_step_id)
        if _name not in catalogue:
            continue
        _params = {
            k: v for k, v in catalogue[_name]['metadata']['parameters'].items()
            if k != 'image'
        }
        _defaults = catalogue[_name]['defaults']
        _existing = _seed.get(_step_id, {})
        _forms[_step_id] = mo.ui.dictionary({
            param_name: make_widget(
                param_spec.get('type', 'string'),
                _existing.get(param_name, _defaults.get(param_name)),
            )
            for param_name, param_spec in _params.items()
        })

    args_form = mo.ui.dictionary(_forms) if _forms else None
    (
        mo.vstack([
            mo.vstack([mo.md(f"**{sid}**"), args_form[sid]])
            for sid in _forms.keys()
        ])
        if args_form is not None
        else mo.md("_No functions in pipeline yet._")
    )
    return (args_form,)


@app.cell
def _(mo):
    apply_args_button = mo.ui.run_button(label="Apply argument changes")
    apply_args_button
    return (apply_args_button,)


@app.cell
def _(apply_args_button, args_form, set_pipeline_args):
    if apply_args_button.value and args_form is not None:
        set_pipeline_args(args_form.value)
    return


@app.cell
def _(catalogue, get_pipeline_args, inspect, mo, pipeline_widget, step_id_to_name):
    _step_ids = pipeline_widget.value.get("value", [])
    if _step_ids:
        _args_map = get_pipeline_args()
        source_accordion = mo.accordion({
            step_id: mo.vstack([
                mo.md(
                    "**Arguments:** " +
                    ", ".join(f"{k}={v}" for k, v in _args_map.get(step_id, {}).items())
                    if _args_map.get(step_id) else "_No arguments._"
                ),
                mo.ui.code_editor(
                    value=inspect.getsource(catalogue[step_id_to_name(step_id)]['function']),
                    language="python",
                    disabled=True,
                ),
            ])
            for step_id in _step_ids
            if step_id_to_name(step_id) in catalogue
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
    file_browser,
    get_pipeline_args,
    mo,
    pipeline_widget,
    run_button,
    run_pipeline,
    step_id_to_name,
):
    mo.stop(not run_button.value, mo.md("Press Run to execute."))
    mo.stop(not file_browser.value, mo.md("Load an image first."))

    _step_ids = pipeline_widget.value.get("value", [])
    mo.stop(not _step_ids, mo.md("No functions in pipeline."))

    _args_map = get_pipeline_args()
    _pipeline = [
        {
            "name": step_id_to_name(sid),
            "args": _args_map.get(sid, catalogue[step_id_to_name(sid)]['defaults']),
        }
        for sid in _step_ids
        if step_id_to_name(sid) in catalogue
    ]

    _image = decode_image(file_browser.path(index=0).read_bytes())
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
                "description": entry['metadata']['description']
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
def _(add_button, catalogue_table, generate_step_id, get_pipeline, set_pipeline):
    if add_button.value and catalogue_table.value:
        selected_names = [row['name'] for row in catalogue_table.value]
        current = get_pipeline()
        new_ids = list(current)
        for name in selected_names:
            sid = generate_step_id(name, new_ids)
            new_ids.append(sid)
        set_pipeline(new_ids)
    return


if __name__ == "__main__":
    app.run()
