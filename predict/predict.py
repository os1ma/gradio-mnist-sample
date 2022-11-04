import numpy
import onnxruntime
from PIL import Image


def preprocess(pil_image):
    # 画像を28*28のサイズに変換
    resized_image = pil_image.resize((28, 28))
    resized_arr = numpy.array(resized_image)

    # Alpha成分だけを抽出
    transposed_arr = resized_arr.transpose(2, 0, 1)
    alpha_arr = transposed_arr[3]

    return alpha_arr


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


if __name__ == '__main__':
    pil_image = Image.open('sample.png')

    input = preprocess(pil_image)

    result = predict(input)
    print(result)
