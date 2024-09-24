import torch
import cv2
import numpy as np
import PIL
import os
from PIL import Image


def metadata_normalize(metadata, base_lon=180, base_lat=90, scale=1000):
    lon, lat = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale

    return torch.tensor([lon, lat])


def convert_binary(image: PIL.Image.Image = None, thr=45, mode="L"):
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    elif isinstance(image, str):
        image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, image_thr = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)
    _, image_thr = cv2.threshold(image, thr, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(image_thr, cv2.COLOR_GRAY2BGR)
    image = Image.fromarray(image).convert(mode)
    return image, image_thr

