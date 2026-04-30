import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import numpy as np
    import cv2
    sys.path.insert(0, '/app')
    from agent.loader import load_catalogue
    from agent.tool_selector import select_tool
    from agent.util import build_image_message

    return build_image_message, cv2, load_catalogue, mo, np, select_tool


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
    ## Ask the Agent
    """)
    return


@app.cell
def _(
    build_image_message,
    catalogue,
    file_upload,
    get_image_sent,
    mo,
    select_tool,
    set_image_sent,
):
    get_result, set_result = mo.state(None)

    # Note config allows temperature, max_tokens and top_p to be set
    def chat_agent(messages, config):
        msgs = list(messages)
        if file_upload.value and not get_image_sent():
            msgs[-1] = build_image_message(messages[-1].content, file_upload)
            set_image_sent(True)
        result = select_tool(msgs, catalogue)
        if result['selected']:
            set_result(result)
        return result['message']

    chat = mo.ui.chat(chat_agent)
    return chat, get_result


@app.cell
def _(chat):
    chat
    return


@app.cell
def _(chat, get_result, mo):
    mo.stop(not chat.value, mo.md("No messages yet."))
    result = get_result()
    mo.stop(result is None, mo.md("No function selected yet."))

    if result.get('reasoning'):
        thinking = mo.accordion({"Model thinking": mo.md(result['reasoning'])})
    else:
        thinking = mo.md("")
    return result, thinking


@app.cell
def _(thinking):
    thinking
    return


@app.cell
def _(mo):
    mo.md("""
    ## Selected Function
    """)
    return


@app.cell
def _(catalogue, mo, result):
    editor = mo.ui.code_editor(
        value=result['source'],
        language="python",
    )
    mo.accordion({
        "Description": mo.md(catalogue[result['name']]['metadata']['description']),
        "Code": editor,
    })
    return (editor,)


@app.cell
def _(mo):
    mo.md("""
    ## Run
    """)
    return


@app.cell
def _(mo, result):
    mo.stop(result is None, mo.md(""))
    mo.md(
        f"Running `{result['name']}` with your uploaded image as the first "
        f"argument. Any edits made in the code editor above will be applied."
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run selected function")
    return (run_button,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(cv2, editor, file_upload, mo, np, result, run_button):
    mo.stop(not run_button.value, mo.md("Press Run to execute."))
    mo.stop(not file_upload.value, mo.md("Upload an image first."))
    mo.stop(result is None, mo.md("No function selected."))

    raw = np.frombuffer(file_upload.value[0].contents, np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    namespace = {}
    # Exec what's in the editor box - allows user to modify function!
    exec(editor.value, namespace)
    fn = namespace[result['name']]
    output = fn(image)

    mo.hstack([
        mo.image(cv2.imencode(".png", image)[1].tobytes(), alt="Original", width=400),
        mo.image(cv2.imencode(".png", output)[1].tobytes(), alt="Output", width=400, vmin=0, vmax=255),
    ])
    return


if __name__ == "__main__":
    app.run()
