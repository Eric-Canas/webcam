"""
_video_webcam.py: Contains the _VideoWebcam class, which is a wrapper for the cv2.VideoCapture class.
This class is used by the Webcam class in webcam.py to read frames from a video file as if it were
a webcam. The wrapper tries to make the interface more similar to the cv2.VideoCapture class, so it
can be used as a drop-in replacement for the cv2.VideoCapture class when working with video files.

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 05-05-2023
"""

import os
import cv2
import time


class _VideoWebcam:
    def __init__(self, video_path: str):
        """
        Initialize the _VideoWebcam.

        :param video_path: str. Path to the video file.
        """
        assert os.path.isfile(video_path), f'Video file not found at {video_path}'
        self.cap = cv2.VideoCapture(video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.start_timestamp = time.time()
        # Just read the first frame to make sure the video is valid
        ret, frame = self.cap.read()
        assert ret, f'Video file at {video_path} is invalid'

    def read(self):
        current_frame = int((time.time() - self.start_timestamp) * self.fps) % self.video_length
        return self.get_required_frame(current_frame)

    def get_required_frame(self, target_frame):
        last_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Calculate the number of frames to skip
        skip_frames = target_frame - last_frame
        if skip_frames < 0:
            skip_frames += self.video_length

        # Use set method if more than 50 frames need to be skipped
        if skip_frames > 50:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.cap.read()
        else:
            # Use grab method to quickly skip the unnecessary frames
            for _ in range(skip_frames - 1):
                self.cap.grab()

            # Retrieve the required frame
            ret, frame = self.cap.retrieve()

        return ret, frame

    def stop(self):
        pass

    def get(self, propId):
        return self.cap.get(propId)

    def set(self, propId, value):
        self.cap.set(propId, value)

    def release(self):
        self.cap.release()

    def isOpened(self):
        # Check if there are more frames to read in the video
        current_frame = int((time.time() - self.start_timestamp) * self.fps)
        return self.cap.isOpened() and current_frame < self.video_length