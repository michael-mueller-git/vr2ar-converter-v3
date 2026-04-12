import gradio as gr
from PIL import Image
import numpy as np
from typing import TypedDict


def combine_masks(im_1: np.ndarray, im_2: np.ndarray):
    """
    Layers images on top of each other, with the foreground image overwriting the background.
    params:
        im_1: background image
        im_2: foreground image
    return:
        combined image
    """
    if im_1.shape != im_2.shape:
        raise ValueError("Images must have the same dimensions")

    # replace rgb values of the background with the foreground where foreground is not fully transparent
    foreground_mask = im_2[:, :, 3] > 0
    im_1[foreground_mask] = im_2[foreground_mask]

    return im_1


def make_grey(image: np.ndarray, grey_value: int = 128):
    """
    Converts opaque pixels within an image to a static grey value.
    params:
        image: RGBA image
    return:
        greyscale image
    """
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    opaque_mask = alpha == 255

    # arbitrary grey value

    # apply grey values to the opaque pixels
    rgb[opaque_mask] = np.stack((grey_value, grey_value, grey_value), axis=-1)

    # combine channels and return
    return np.dstack((rgb, alpha))
    
class ImageData(TypedDict):
    background: np.ndarray
    layers: list[np.ndarray]


def mask(image: ImageData):
    print("apply")
    print(image)
    bg = image.get("composite", None)

    return [bg, bg]


with gr.Blocks() as demo:
    """
    imageMask = gr.ImageMask(
        type="pil",
        label="Input Image",
        layers=False,
        sources=["upload"],
        interactive=True,
        width=800,
        height=800,
    )
        """
    imageMask = gr.ImageEditor(
        type="pil",
        label="Input Image",
        layers=False,
        sources=["upload"],
        interactive=True,
        width=800,
        height=800,
    )
    imageMask2 = gr.ImageEditor(
        type="pil",
        label="Input Image",
        layers=False,
        sources=["upload"],
        interactive=True,
        width=800,
        height=800,
    )
    im_1 = gr.Image()
    im_2 = gr.Image()
    imageMask.apply(mask, inputs=imageMask, outputs=[im_1, im_2])
    imageMask2.apply(mask, inputs=imageMask2, outputs=[im_1, im_2])

if __name__ == "__main__":
    demo.launch()
