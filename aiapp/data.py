from urllib.request import urlopen
from PIL import Image


def read_image(image_path: str):
    """
    image_path: (str) path or url
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(urlopen(image_path))
    else:
        image = Image.open(image_path)