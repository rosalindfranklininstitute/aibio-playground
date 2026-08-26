# Getting started page
This is a tutorial...

marimo-backed python blocks work:
```python {marimo}
2 + 2
```

/// html | div
Basic html blocks work
///

embeded blocks appear to work only if `mode: read` and `app_width: compact` or `app_width: full`

/// marimo-embed
    height: 300px
    mode: read
    app_width: compact

```python
@app.cell
def __():
    import marimo as mo

    slider = mo.ui.slider(1, 10, value=5)
    slider
    return

@app.cell
def __():
    mo.md(f"Selected value: **{slider.value}**")
    return
```

///


/// marimo-embed
    height: 400px
    mode: read
    app_width: full

```python
@app.cell
def __():
    import marimo as mo

    name = mo.ui.text(placeholder="Enter your name", debounce=False)
    name
    return

@app.cell
def __():
    mo.md(f"Hello, **{name.value or '__'}**!")
    return
```

///
