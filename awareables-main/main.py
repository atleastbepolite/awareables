"""
Image capture
USB cam / IMX219 -> opencv.ImRead -> jpg
Pre-processing
jpg -> Filters -> opencv image
opencv image -> Image segmentation -> jpgs
Classification
Directory of numbered jpgs -> Classifier
Prediction -> Array
Array -> liblouis (Braille translator) -> String
Post-processing
String -> Spellcheck/correction -> String
String -> Google API -> wav
Audio outt
wav -> speaker
"""

import os, sys, argparse, shutil, time
import cv2
from abc import abstractmethod
from preprocess_1210 import preProcess
from classifier import Classifier
from postProcessing.SpellChecker import SpellChecker
from usb_camera_simple import show_camera
from postProcessing.textToSpeech import TextToSpeech
from PIL import Image
from AngelinaReader.crop import cropFrom
from cursor import openFingerCursor


if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser(
        prog = "Aware-ables",
        description = "Braille Detection Pipeline")
    parser.add_argument('-m', '--ml', dest='mlCrop', action='store_true', default=False, required=False)
    parser.add_argument('-s', '--setting', dest='cropSetting', choices=[0, 1, 2], default=0, required=False, type=int)
    parser.add_argument('-c', '--cursor', dest='fingerCursor', action='store_true', default=False, required=False)
    parser.add_argument('-i', '--image', dest='inputImage', default=None, required=False, type=str)
    config = parser.parse_args()
    
    # start submodules
    sc = SpellChecker('postProcessing/txtFiles/dict_text.txt')
    tts = TextToSpeech()
    c = Classifier()

    # wait for button
    while True:
        print("Running with settings:")
        if config.mlCrop:
            print("Machine Learning Crop")
        else:
            print("Fast Crop")
        if config.inputImage is not None:
            print(f"Input Image: {config.inputImage}")
        if config.fingerCursor:
            print("Finger Cursor: TRUE")
        
        # clear cropped directory
        for filename in os.listdir('/home/b1-awareables/Documents/awareables/cropped'):
            file_path = os.path.join('/home/b1-awareables/Documents/awareables/cropped', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s: Reason %s" % (file_path, e))
        
        # capture / open image
        if config.inputImage is None:
            satisfied = False
            while not satisfied:
                show_camera()
                Image.open('capture.jpg').show()
                satisfied = (input("image aligned [y/n]? ") == "y")
            image = 'capture.jpg'
            image = os.path.join(os.getcwd(), image)
        else:
            image = config.inputImage
        
        start = time.time()
        # preprocessing
        print('Beginning preprocessing phase (1/3)...')
        cropdir = '/home/b1-awareables/Documents/awareables/cropped'
        meta = None
        if config.mlCrop:
            result = cropFrom(image, cropdir)
            coords = result["crops"]
            meta = {"homography": result["homography"], "image": result["image"]}
        else:
            coords = preProcess(image, cropdir, config.cropSetting)
        print(coords)
        
        # classification
        print('Beginning classification phase (2/3)...')
        raw_string = c.run(cropdir)
        print('Classifier output: ' + raw_string[0], raw_string[2])
        
        # post-processing
        print('Beginning post-processing phase (3/3)...')
        final_string = sc.wordIntake(raw_string[0], confidenceMatrix=raw_string[1])
        print('Spellcheck result: ' + final_string)
        
        end = time.time()
        tts.synthesize_speech(final_string)
        print(f"capture-to-read completed in {end-start}s")
        
        # finger-cursor
        if config.fingerCursor:
            openFingerCursor(image, coords, raw_string[2], (config.inputImage == None), meta)
        
        # reset flags
        config.inputImage = None
        
        new_args = input("new args: ")
        if len(new_args) > 0:
            config = parser.parse_args(new_args.split(' '))
