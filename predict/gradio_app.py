import gradio as gr

from predict import predict

app = gr.Interface(title="Handwritten Number Prediction",
                   fn=predict, inputs="sketchpad", outputs="label",
                   live=True, allow_flagging="never",
                   css=".gradio-container { max-width: 800px; margin: 0 auto; }")

app.launch()
