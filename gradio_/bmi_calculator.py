import gradio as gr
from librosa import ex


def calc_bmi(weight, height):
    try:
        weight, height = float(weight), float(height)
        bmi = weight / (height/100)**2
    except ValueError:
        return "Please enter valid numbers"
    except ZeroDivisionError:
        return "Please enter a non-zero height"
    return bmi


examples = [
    ["80", "180"],
    ["60", "170"],
    ["100", "190"],
    ["50", "150"],
    ["57", "168"]
]

description = "This is a simple BMI calculator. Enter your weight in kilograms and your height in centimeters."
article = """BMI is calculated by dividing a person's weight (in kilograms) by the square of their height (in meters).

A BMI between 18.5 and 25 is considered healthy, less than 18.5 is underweight, and over 25 is overweight.
"""

demo = gr.Interface(fn=calc_bmi, inputs=[
                    "text", "text"], outputs="text", title="BMI Calculator", description=description,
                    article=article, theme="legacy", examples=examples, live=True, allow_flagging=False)

demo.launch()
