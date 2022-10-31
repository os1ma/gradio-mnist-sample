import gradio as gr
import numpy
import onnxruntime


def predict(image):
    reshaped_arr = image.reshape(-1)
    print("reshaped_arr")
    print(reshaped_arr)

    input = [reshaped_arr.astype(numpy.float32)]
    print("input")
    print(input)

    # predict

    onnx_session = onnxruntime.InferenceSession("../train/model.onnx")

    output = onnx_session.run(['probabilities'], {'float_input': input})
    print("output")
    print(output)

    result = output[0][0]
    print("result")
    print(result)

    name = "world"
    return "Hello " + name + "!"

demo = gr.Interface(fn=predict, inputs="sketchpad", outputs="text")

demo.launch()
