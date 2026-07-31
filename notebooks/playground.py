import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import base64, html, inspect, sys, traceback
    from wigglystuff import SortableList, CellTour, WidgetDAG
    sys.path.insert(0, '/app')
    from agent.catalogue_loader import load_catalogue
    from agent.pipeline_builder import step, ConversationState, get_client, run_pipeline, generate_step_id, step_id_to_name
    from agent.image_tools import encode_png
    from agent.image_loader import inspect_image, load_image, SUPPORTED_EXT, downsample_for_png

    return (
        mo,
        base64,
        html,
        inspect,
        traceback,
        SortableList,
        CellTour,
        WidgetDAG,
        load_catalogue,
        step,
        ConversationState,
        get_client,
        run_pipeline,
        generate_step_id,
        step_id_to_name,
        encode_png,
        inspect_image,
        load_image,
        SUPPORTED_EXT,
        downsample_for_png,
    )


@app.cell
def _(CellTour, mo):
    tour = mo.ui.anywidget(
        CellTour(
            steps=[
                {
                    "cell_name": "upload_image_step",
                    "title": "1. Upload an image",
                    "description": "Upload an image to get started. If the image is 3D, a time series, or has multiple channels, you will be able to select a 2D slice and channel(s).",
                },
                {
                    "cell_name": "ask_agent_step",
                    "title": "2. Ask the agent",
                    "description": "Tell the agent what you want to achieve. It will ask clarifying questions before proposing a pipeline.",
                },
                {
                    "cell_name": "pipeline_sortablelist",
                    "title": "3. View the pipeline",
                    "description": "The proposed analysis pipeline will appear here. You can reorder or remove functions if needed.",
                },
                {
                    "cell_name": "run_pipeline_step",
                    "title": "4. Run the pipeline",
                    "description": "Press run to apply the pipeline to your image data. The result appears in a box below."
                },
                {
                    "cell_name": "available_functions_step",
                    "title": "5. Add extra functions",
                    "description": "If the agent missed something, you can select additional functions to add to the pipeline here.",
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
def _(SUPPORTED_EXT, new_conversation, mo):
    upload = mo.ui.file(
        filetypes=SUPPORTED_EXT,
        kind="button",
        label="Upload a microscopy image",
        max_size=250000000,
        on_change=new_conversation,
    )
    return (upload,)


@app.cell
def upload_image_step(upload):
    upload
    return


@app.cell
def _(inspect_image, mo, upload):
    mo.stop(not upload.value, None)
    file_bytes = upload.value[0].contents
    fname = upload.value[0].name
    info = inspect_image(file_bytes, fname)
    return fname, file_bytes, info


@app.cell
def _(fname, info, mo):
    mo.md(f"Uploaded {fname}.\n\n**Dims:** `{info['dims']}` {info['shape']}")
    return

@app.cell
def _(info, mo):
    mo.accordion({"Loaded metadata": mo.ui.table(info["metadata"], selection=None)})
    return

@app.cell
def _(info, mo):
    _dim_size = dict(zip(info["dims"], info["shape"]))

    dims_input = mo.ui.text(value=info["dims"], label="Re-order Axes:")

    t_enabled = _dim_size.get("T", 1) > 1
    z_enabled = _dim_size.get("Z", 1) > 1
    channel_dim = "S" if "S" in info["dims"] else ("C" if "C" in info["dims"] else None)
    channel_enabled = channel_dim is not None and _dim_size.get(channel_dim, 1) > 1
    channel_stop = max(_dim_size.get(channel_dim, 1) - 1, 0) if channel_dim else None

    t_input = mo.ui.number(
        start=0, step=1, stop=max(_dim_size.get("T", 1) - 1, 0),
        value=0, label="T-index", disabled=not t_enabled,
    )
    z_input = mo.ui.number(
        start=0, step=1, stop=max(_dim_size.get("Z", 1) - 1, 0),
        value=0, label="Z-index", disabled=not z_enabled,
    )
    return channel_enabled, channel_stop, dims_input, t_enabled, t_input, z_enabled, z_input

@app.cell
def _(channel_enabled, mo):
    channel_none = mo.ui.checkbox(
        value=True,
        label="Composite (selects up to first three channels)",
        disabled=not channel_enabled,
    )
    return (channel_none,)

@app.cell
def _(channel_enabled, channel_none, channel_stop, mo):
    channel_input = mo.ui.number(
        start=0, step=1, stop=channel_stop, value=0, label="Selected channel",
    ) if (channel_enabled and not channel_none.value) else None
    return (channel_input,)

@app.cell
def _(channel_input, channel_none, dims_input, mo, t_input, z_input):
    mo.vstack([
        mo.md("### Parameters (advanced usage)"),
        dims_input, t_input, z_input, channel_none,
        channel_input if channel_input is not None else mo.md("_Composite_"),
    ])
    return

@app.cell
def _(mo, upload):
    reload_button = mo.ui.run_button(label="Reload image", disabled=not upload.value)
    reload_button
    return (reload_button,)

@app.cell
def _(mo):
    get_loaded_image, set_loaded_image = mo.state(None)
    return get_loaded_image, set_loaded_image

@app.cell
def _(fname, file_bytes, load_image, set_loaded_image):
    _png, _array = load_image(file_bytes, fname=fname)
    set_loaded_image({'png': _png, 'array': _array})
    return

@app.cell
def _(
    channel_enabled,
    channel_input,
    channel_none,
    dims_input,
    fname,
    file_bytes,
    info,
    load_image,
    reload_button,
    set_loaded_image,
    t_enabled,
    t_input,
    z_enabled,
    z_input,
):
    if reload_button.value:
        _dims_override = dims_input.value if dims_input.value != info["dims"] else None
        _png, _array = load_image(
            file_bytes,
            fname=fname,
            dims_override=_dims_override,
            t=int(t_input.value) if t_enabled else 0,
            z=int(z_input.value) if z_enabled else None,
            channel=int(channel_input.value) if (channel_enabled and not channel_none.value) else None,
        )
        set_loaded_image({'png': _png, 'array': _array})
    return

@app.cell
def _(get_loaded_image, mo):
    _loaded = get_loaded_image()
    (
        mo.vstack([mo.md("### Image to be processed"), mo.image(_loaded['png'], width=400)])
        if _loaded else mo.md("_Upload an image first_")
    )
    return

@app.cell
def ask_agent_step(mo):
    mo.md("## Pipeline Agent")
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
    get_reasoning, set_reasoning = mo.state(None)
    return get_reasoning, set_reasoning


@app.cell
def _(
    base64,
    catalogue,
    client,
    generate_step_id,
    get_conv_state,
    get_image_sent,
    get_loaded_image,
    mo,
    set_conv_state,
    set_image_sent,
    set_pipeline,
    set_pipeline_args,
    set_reasoning,
    step,
    upload,
):
    def chat_agent(messages, config):
        image_b64 = None
        if not get_image_sent():
            _loaded = get_loaded_image()
            if _loaded:
                image_b64 = base64.b64encode(_loaded['png']).decode('utf-8')
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
        disabled=not upload.value,
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
    def make_widget(param_type, value, label=None):
        if param_type == "number":
            return mo.ui.number(value=value, label=label)
        elif param_type == "boolean":
            return mo.ui.checkbox(value=bool(value), label=label)
        else:
            return mo.ui.text(value=str(value), label=label)
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
            if k != 'image_data'
        }
        _defaults = catalogue[_name]['defaults']
        _existing = _seed.get(_step_id, {})
        _forms[_step_id] = mo.ui.dictionary({
            param_name: make_widget(
                param_spec.get('type', 'string'),
                _existing.get(param_name, _defaults.get(param_name)),
                label=param_name,
            )
            for param_name, param_spec in _params.items()
        })

    args_form = mo.ui.dictionary(_forms) if _forms else None
    return (args_form,)

@app.cell
def _(args_form, catalogue, html, mo, step_id_to_name):
    if args_form is None:
        display = mo.md("_No functions in pipeline yet_")
    else:
        _blocks = []
        for _step_id, _step_widgets in args_form.items():
            _param_specs = catalogue[step_id_to_name(_step_id)]['metadata']['parameters']
            _rows = []
            for param_name, widget in _step_widgets.items():
                desc = html.escape(_param_specs.get(param_name, {}).get('description', ''))
                _rows.append(mo.hstack(
                    [widget, mo.md(f'<span title="{desc}" style="font-size:1.3em;">ⓘ</span>')],
                    justify='start', gap=1,
                ))
            if not _rows:
                _rows = [mo.md("_No arguments._")]
            _blocks.append(mo.vstack([mo.md(f"**{_step_id}**"), *_rows]))
        display = mo.vstack(_blocks)
    display
    return

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
        if _names else "_No functions in pipeline yet_"
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
    encode_png,
    downsample_for_png,
    get_loaded_image,
    get_pipeline_args,
    mo,
    pipeline_widget,
    run_button,
    run_pipeline,
    step_id_to_name,
    traceback,
    upload,
):
    mo.stop(not run_button.value, mo.md(""))
    mo.stop(not upload.value, mo.md("Load an image first."))

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

    _image = get_loaded_image()['array']
    _image_data = {'source': _image, 'current': _image, 'info': {}}

    _error = None
    _result_data = None
    _history = []

    def _record(name, data):
        _thumb = downsample_for_png(data['current'], max_dim=250)
        _history.append((name, encode_png(_thumb)))

    with mo.capture_stdout() as _stdout_buf, mo.capture_stderr() as _stderr_buf:
        try:
            _result_data = run_pipeline(_image_data, _pipeline, catalogue, on_step=_record)
        except Exception:
            _error = traceback.format_exc()

    _log_text = _stdout_buf.getvalue() + _stderr_buf.getvalue()
    _logs = mo.accordion({"Pipeline logs": mo.md(f"```\n{_log_text or '(no output)'}\n```")})

    if _error is not None:
        result_display = mo.vstack([
            mo.md(f"**Pipeline failed:**\n```\n{_error}\n```"),
            _logs,
        ])
    else:
        _blocks = [
            mo.image_compare(
                encode_png(_result_data['source']),
                encode_png(_result_data['current']),
                width=400,
                height=400,
            ),
        ]
        if _result_data.get('info'):
            _blocks.append(mo.accordion({"Additional info": mo.ui.table(
                [{'key': k, 'value': v} for k, v in _result_data['info'].items()],
                selection=None,
            )}))
        _blocks.append(_logs)
        result_display = mo.vstack(_blocks, align="center")
    history = _history
    results_display
    return (history,)

@app.cell
def _(WidgetDAG, history, mo):
    _NODES_PER_ROW = 6

    def _caption(name):
        return "Source" if name == "source" else f"After {name}"

    def _rows(hist, nodes_per_row=_NODES_PER_ROW):
        return [hist[i:i + nodes_per_row] for i in range(0, len(hist), nodes_per_row)]

    def _row_widget(row):
        _seen = set()
        _nodes = {}
        for name, png in row:
            _cap = _caption(name)
            while _cap in _seen:  # triggers on a duplicate name
                _cap += "\u200b"
            _seen.add(_cap)
            _nodes[_cap] = mo.image(png, width=250)
        _ids = list(_nodes.keys())
        _edges = [(_ids[i], _ids[i + 1]) for i in range(len(_ids) - 1)]
        return WidgetDAG(nodes=_nodes, edges=_edges)

    pipeline_graph = (
        mo.vstack([_row_widget(_row) for _row in _rows(history)])
        if history else mo.md("_Run the pipeline to see the step graph._")
    )
    pipeline_graph
    return

@app.cell
def _(mo):
    mo.md("## Additional Functions")
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
        label="Functional Catalogue",
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
