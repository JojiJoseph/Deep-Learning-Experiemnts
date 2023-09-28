import numpy as np
import gradio as gr
from traitlets import default

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

def invert(input_img):
    return 1 - input_img

def grayscale(input_image):
    grayscale_filter = np.array([
        [0.3, 0.6, 0.1], 
        [0.3, 0.6, 0.1], 
        [0.3, 0.6, 0.17]
    ])
    grayscale_img = input_image.dot(grayscale_filter.T)
    grayscale_img = np.clip(grayscale_img, 0, 255).astype(np.uint8)
    return grayscale_img

def solarize(input_img):
    return np.where(input_img < 128, input_img, 255 - input_img)

def apply_filter(input_img, filter):
    if filter == "sepia":
        return sepia(input_img)
    elif filter == "grayscale":
        return grayscale(input_img)
    elif filter == "invert":
        return invert(input_img)
    elif filter == "solarize":
        return solarize(input_img)
    else:
        return input_img


demo = gr.Interface(apply_filter, [gr.Image(label="input image"), gr.Dropdown(["original","sepia","grayscale","invert", "solarize"])], "image", live=True)
demo.launch()
