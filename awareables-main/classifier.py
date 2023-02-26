import os, time, re
import louis
from onnx_predict import make_prediction

class_labels = [
    "⠀",
    "⠁", #a
    "⠂", #,
    "⠃", #b
    "⠄", #'
    "⠅", #k
    "⠆", #;
    "⠇", #L
    "⠉", #c
    "⠊", #i
    "⠋", #f
    "⠍", #m
    "⠎", #s
    "⠏", #p
    "⠑", #e
    "⠒", #-
    "⠓", #h
    "⠕", #o
    "⠖", #!
    "⠗", #r
    "⠙", #d
    "⠚", #j
    "⠛", #g
    "⠝", #n
    "⠞", #t
    "⠟", #q
    "⠠", #capital
    "⠤", #underscore
    "⠥", #u
    "⠦", #"
    "⠧", #v
    "⠭", #x
    "⠲", #.
    "⠵", #z
    "⠺", #w
    "⠼", #number
    "⠽", #y
]

def numFromString(s):
    if s.endswith('.jpg'):
        return int(re.findall(r'\d+', s)[0])
    else:
        return -1

class Classifier():
    def __predict(self, dir):
        matrix = {}
        try:
            start = time.time()
            prediction = make_prediction(dir)
            end = time.time()
        except Exception as e:
            print(e)
            return None
        # print(prediction)
        brl_list = [class_labels[i[0][0]] for i in prediction]
        braille = ''.join(brl_list)
        # print(braille)
        for i in range(len(prediction)):
            entry = prediction[i]
            if entry[1] < 0.9 and entry[0][0] != 0:
                matrix[i] = [louis.backTranslateString(["unicode.dis", "en-ueb-g1.ctb"], class_labels[e]) for e in entry[0][:5]]
        
        return (louis.backTranslateString(["unicode.dis", "en-ueb-g1.ctb"], braille), matrix, brl_list)
        
    def __init__(self):
        pass

    def run(self, dir):
        print("Running prediction on " + dir)
        return self.__predict(dir)

if __name__ == '__main__':
    c = Classifier()
    userdir = input('Input directory: ')
    inputdir = os.path.join(os.getcwd() + '/' + userdir)
    raw_string = c.run(inputdir)
    print('Classifier output: ')
    print(raw_string)
