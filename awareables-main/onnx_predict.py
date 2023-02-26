# predict.py

import onnxruntime as ort
import numpy as np
import cv2, os, re

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

# Load the network into an MXNet module and bind the corresponding parameters
mod = ort.InferenceSession('models/filtered18+braille.onnx', providers=['TensorrtExecutionProvider','CPUExecutionProvider'])
print("warming up...")

for i in range(5):
    prob = mod.run(None, {mod.get_inputs()[0].name: np.zeros((10, 3, 28, 28), dtype=np.float32)})
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    
def numFromString(s):
    if s.endswith('.jpg'):
        find = re.findall(r'\d+', s)
        if (len(find) == 0):
            return -1
        return int(find[0])
    else:
        return -1

# returns order list of inferences and highest confidence
def predict(dirname, mod):
    data = np.empty((10, 3, 28, 28), dtype = np.float32)
    idx = 0
    result = []
    for root, dirs, files in os.walk(dirname):
        files.sort(key=numFromString)
        #files.sort()
        for filename in files:
            if filename.endswith('.jpg') and (filename.find('labeled') == -1 and filename.find('marked') == -1):
                #print(filename)
                img = cv2.imread(os.path.join(root, filename))
                if img is None:
                    print("lost")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (28, 28))
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 2)
                data[idx] = img
                idx += 1
                
                if (idx == 10):
                    output = mod.run(None, {mod.get_inputs()[0].name: data})
                    for i in range(idx):
                        prob = np.squeeze(output[0][i])
                        a = np.argsort(prob)[::-1]
                        #print(a, prob[a[0]])
                        result.append((a, prob[a[0]]))
                    idx = 0
    if idx != 0:
        output = mod.run(None, {mod.get_inputs()[0].name: data})
        for i in range(idx):
            prob = np.squeeze(output[0][i])
            a = np.argsort(prob)[::-1]
            #print(a, prob[a[0]])
            result.append((a, prob[a[0]]))
    return result

# Code to predict on a local file
def make_prediction(dirname):
    return predict(dirname, mod)

if __name__=='__main__':
    print(''.join([class_labels[i[0][0]] for i in make_prediction('lullabye')]))
