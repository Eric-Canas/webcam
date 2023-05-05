"""
_webcam_background.py: Contains the _WebcamBackground class, which is a wrapper for the imutils
WebcamVideoStream class. It is used by the Webcam class in webcam.py to read frames from a webcam
running in a background thread. This wrapper tries to make the interface more similar to the
cv2.VideoCapture class, so it can be used as a drop-in replacement for the cv2.VideoCapture class.

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 05-05-2023
"""

from __future__ import annotations
from imutils.video import WebcamVideoStream


class _WebcamBackground(WebcamVideoStream):
    def __init__(self, src=0, name="WebcamVideoStream"):
        super().__init__(src=src, name=name)

    def read(self):
        frame = super().read()
        return self.grabbed, frame

    def stop(self):
        super().stop()
        self.grabbed = False

    def get(self, propId):
        return self.stream.get(propId=propId)

    def set(self, propId, value):
        self.stream.set(propId=propId, value=value)

    def release(self):
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.release()

    def __del__(self):
        self.stop()
        self.release()

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()
