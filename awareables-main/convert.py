import mxnet as mx
import numpy as np
import cv2
from PIL import Image
from matplotlib.pyplot import imshow

import logging
class_labels = [
    "⠀",
    "⠁",
    "⠂",
    "⠃",
    "⠄",
    "⠅",
    "⠆",
    "⠇",
    "⠉",
    "⠊",
    "⠋",
    "⠍",
    "⠎",
    "⠏",
    "⠑",
    "⠒",
    "⠓",
    "⠕",
    "⠖",
    "⠗",
    "⠙",
    "⠚",
    "⠛",
    "⠝",
    "⠞",
    "⠟",
    "⠠",
    "⠤",
    "⠥",
    "⠦",
    "⠧",
    "⠭",
    "⠲",
    "⠵",
    "⠺",
    "⠼",
    "⠽",
]
logging.basicConfig(level=logging.INFO)
 
sym = './image-classification-symbol.json'
params = './image-classification-0015.params'
in_shapes = [(10, 3, 28, 28)]
in_types = [np.float32]
onnx_file = './model.onnx'
converted_model_path = mx.onnx.export_model(sym, params, in_shapes, in_types, onnx_file)
mx.viz.plot_network(mx.symbol.load(sym), node_attrs={"shape":"oval", "fixedsize":"false"})

print("all done")

from onnx import checker
import onnx

model_proto = onnx.load_model(converted_model_path)

checker.check_graph(model_proto.graph)

import onnxruntime as ort

input_data = np.empty((10, 3, 28, 28), dtype=np.float32)
for i in range(6):
	img = cv2.imread('filtered/' + str(i+1) + '.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (28, 28))
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 1, 2)
	input_data[i] = img
print(input_data)
ort_sess = ort.InferenceSession('model.onnx')
print(ort_sess.get_inputs()[0].name)
outputs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: input_data})
print(outputs)
for i in range(10):
	prob = np.squeeze(outputs[0][i])
	a = np.argsort(prob)[::-1]
	print (class_labels[a[0]], prob[a[0]])
