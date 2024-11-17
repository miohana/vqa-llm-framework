import base64
import requests
from PIL import Image
from transformers.image_utils import load_image


def load_image_as_base64(image_url: str) -> str:
    image_data = base64.b64encode(
        requests.get(image_url).content
    ).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"


def load_image_as_pil(image_path_or_url: str) -> Image.Image:
    return load_image(image_path_or_url)
