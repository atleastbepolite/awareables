#!/usr/bin/env python3
#
#  USB Camera - Simple
#
#  Copyright (C) 2021-22 JetsonHacks (info@jetsonhacks.com)
#
#  MIT License
#

import sys
import RPi.GPIO as GPIO
from PIL import Image
import cv2
window_title = "USB Camera"

but_pin = 18  # Board pin 18

GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup(but_pin, GPIO.IN)  # button pin set as input
def show_camera():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video0"
    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    """ 
    # How to set video capture properties using V4L2:
    # Full list of Video Capture Properties for OpenCV: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #Select Pixel Format:
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    # Two common formats, MJPG and H264
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # Default libopencv on the Jetson is not linked against libx264, so H.264 is not available
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    # Select frame size, FPS:
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    """
    if video_capture.isOpened():
        try:
            #window_handle = cv2.namedWindow(
            #    window_title, cv2.WINDOW_AUTOSIZE )
            # Window
            prev_value = GPIO.input(but_pin)
            curr_value = GPIO.input(but_pin)
            while True:
                prev_value = curr_value
                ret_val, frame = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                '''
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                cv2.waitKey(10)
                '''
                curr_value = GPIO.input(but_pin)
                if curr_value != prev_value and curr_value == 0:
                    cv2.imwrite('./capture.jpg', frame)
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":

    show_camera()
