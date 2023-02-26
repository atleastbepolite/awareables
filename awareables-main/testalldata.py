import sys, os, csv
import louis, time
from onnx_predict import make_prediction

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

print("starting predictions...")
i = 0
start = time.time()
for root, dirs, files in os.walk('../data'):
#for root, dirs, files in os.walk('/home/kevin/Documents/filtered'):
    for d in dirs:
        print(d)
        if not (root.endswith('rejected')):
            actual = root[-1]
            try:
                # print("predicting " + file)
                prediction = make_prediction(os.path.join(root, d))
            except:
                continue
            i = len(files)
            break
end = time.time()
print(f"made {i} predictions in {end-start} seconds")
