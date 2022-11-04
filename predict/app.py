import gradio as gr
import numpy
import onnxruntime


def predict(image):
    if image is None:
        return None

    # 2次元配列を1次元に変換
    reshaped_arr = image.reshape(-1)
    # ONNX Runtimeへの入力形式に変換
    input = [reshaped_arr.astype(numpy.float32)]

    # モデルの読み込み
    onnx_session = onnxruntime.InferenceSession("../train/model.onnx")
    # 推論
    output = onnx_session.run(['probabilities'], {'float_input': input})

    # 確率の配列を取り出し
    probabilities = output[0][0]
    # Dictに変換
    result = {}
    for i, probability in enumerate(probabilities.tolist()):
        result[i] = probability

    return result


app = gr.Interface(title="Handwritten Number Prediction", fn=predict, inputs="sketchpad", outputs="label", live=True,
                   allow_flagging="never", css=".gradio-container { max-width: 800px; margin: 0 auto; }")

app.launch()
