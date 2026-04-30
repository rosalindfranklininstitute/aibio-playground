import base64


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
