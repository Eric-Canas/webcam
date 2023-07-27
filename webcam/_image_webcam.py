"""
_image_webcam.py: Contains the _ImageWebcam class, which is a convenient class that shares interface with
cv2.VideoCapture.
This class is used by the Webcam class in webcam.py to simulate a webcam stream from a single image.
It is a webcam that will always retrieve the same image, and will always return True when read() is called,
except when the image is closed (when release() is called).

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 27-07-2023
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import time


class _ImageWebcam:
    def __init__(self, image_source: str|np.ndarray, _dummy_fps: int = 30):
        """
        Initialize the _ImageWebcam.

        :param image_source: str or np.ndarray. Path to the image file or a numpy array of image
        """
        if isinstance(image_source, str):
            assert os.path.isfile(image_source), f'Image file not found at {image_source}'
            self.img = cv2.imread(image_source)
        elif isinstance(image_source, np.ndarray):
            self.img = image_source
        else:
            raise ValueError('image_source must be a file path or a numpy array')

        self.image_width = self.img.shape[1]
        self.image_height = self.img.shape[0]

        self.fps = _dummy_fps

        self.start_timestamp = time.time()

    def read(self):
        if self.isOpened():
            assert isinstance(self.img, np.ndarray), 'Non image available, but isOpened() is True'
            return True, self.img.copy()
        else:
            return False, None

    def stop(self):
        self.release()

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.image_width
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.image_height
        elif propId == cv2.CAP_PROP_FPS:
            return self.fps
        else:
            return None

    def set(self, propId, value):
        pass

    def release(self):
        self.img = None

    def isOpened(self):
        return self.img is not None
