"""
webcam.py: Contains the Webcam class that provides an interface for reading frames from a webcam.
The class allows specifying various parameters such as frame size, color format, batch size, and the maximum
frame rate. It can also run in the background for faster frame reading.

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 04-03-2023
"""


from __future__ import annotations
import cv2
import numpy as np
import time
from typing import Tuple

import os

from webcam._video_webcam import _VideoWebcam
from webcam._webcam_background import _WebcamBackground

CROP, RESIZE = 'crop', 'resize'

class Webcam:
    def __init__(
        self,
        webcam_src: int | str = 0,
        h: int | None = None,
        w: int | None = None,
        as_bgr: bool = False,
        batch_size: int | None = None,
        run_in_background: bool = False,
        max_frame_rate: int | None = 60,
        on_aspect_ratio_lost: str = CROP,
    ):
        """
        Initialize the WebcamReader.

        :param webcam_src: int or str. The index of the webcam or its path.
        :param h: int or None. Desired height of the frames. If None and `_w` is provided, the aspect ratio will be preserved.
        :param w: int or None. Desired width of the frames. If None and `_h` is provided, the aspect ratio will be preserved.
        :param as_bgr: bool. If True, the frames will be returned in BGR format, otherwise in RGB format.
        :param batch_size: int or None. If not None, the iterator will yield batches of frames (B, H, W, C). If None, the iterator will yield single frames (H, W, C).
        :param run_in_background: bool. If True, the frames will be read in a background thread (speeding up the reading).
        :param max_frame_rate: int or None. The maximum frame rate to read the frames at. If None, there will be no limitations on frame rating.
        """
        assert on_aspect_ratio_lost in (CROP, RESIZE), f"Invalid value for `on_aspect_ratio_lost`: {on_aspect_ratio_lost}." \
                                                       f" Valid values are: {CROP}, {RESIZE}"
        self._background = run_in_background
        self.on_aspect_ratio_lost = on_aspect_ratio_lost
        if run_in_background:
            raise NotImplementedError("Background reading is not implemented yet.")
            #self.cap = _WebcamBackground(src=webcam_src).start()
        elif isinstance(webcam_src, str):
            #TODO: Improve the check for video file
            assert os.path.isfile(webcam_src), f"Video file not found: {webcam_src}"
            self.cap = _VideoWebcam(video_path=webcam_src)
        else:
            self.cap = cv2.VideoCapture(webcam_src)
        self.as_bgr = as_bgr

        # Calculate and set output frame size
        self.frame_size_hw = self._calculate_and_set_desired_resolution(h, w)

        # Set batch size and frame rate attributes
        self.batch_size = batch_size
        self.start_timestamp = time.time()
        self.max_frame_rate = max_frame_rate
        self.last_frame_timestamp = self.start_timestamp

    @property
    def _h(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def _w(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def h(self) -> int:
        return self.frame_size_hw[0]

    @property
    def w(self) -> int:
        return self.frame_size_hw[1]

    @property
    def current_timestamp_seconds(self):
        return time.time() - self.start_timestamp

    def read_next_frame(self) -> np.ndarray:
        """
        Read the next frame from the video. (Skipping the frames_offset if it is greater than one.)
        """
        batch_size = 1 if self.batch_size is None else self.batch_size
        frames = []
        for i in range(batch_size):
            ret, frame = self.read()
            if not ret:
                raise StopIteration("The webcam have closed?.")
            frames.append(frame)

        if self.max_frame_rate is not None:
            # Sleep to simulate the frame rate (0.8 is a magic number, just to compensate the execution resuming time)
            time.sleep(max(0, (1/self.max_frame_rate)*0.8 - (time.time() - self.last_frame_timestamp)))
            self.last_frame_timestamp = time.time()

        return frames[0] if self.batch_size is None else np.stack(frames, axis=0)


    def read_video_iterator(self, from_start: bool = False) -> np.ndarray:
        """
        Yields the frames of the video. If simulate_frame_rate is not None,
         the frames are yielded at the given simulated frame rate. That is, skipping real_frame_rate / simulate_frame_rate
         at each iteration.

        :param from_start: If True, the iterator will start from the beginning of the video.

        :return: The next frame in the video.
        """
        # While webcam is open
        while self.cap.isOpened():
            yield self.read_next_frame()
        raise StopIteration

    def _calculate_and_set_desired_resolution(self, h: int | None = None, w: int | None = None) -> tuple[int, int]:
        """
        Calculate and set the optimal webcam resolution based on the desired width and height.
        The resolution will only change if the new resolution is natively supported by the webcam
        and maintains the maximum available aspect ratio.

        :param h: int or None. Desired height of the frames.
        :param w: int or None. Desired width of the frames.
        :return: tuple. The final frame size (height, width).
        """
        # Set webcam to its maximum supported resolution
        max_h, max_w = self._set_webcam_resolution(h=1e6, w=1e6)

        if h is None and w is None:
            return max_h, max_w

        # Calculate the missing dimension while keeping the aspect ratio if only one dimension is provided
        if h is None or w is None:
            h, w = self.calculate_frame_size_keeping_aspect_ratio(h=h, w=w)

        # Change the resolution if supported (and keeps the maximum aspect ratio if only one dimension was provided)
        if self._is_resolution_natively_supported(h=h, w=w):
            self._set_webcam_resolution(h=h, w=w)

        return h, w

    def _set_webcam_resolution(self, h: int, w: int) -> Tuple[int, int]:
        """
        Set the webcam resolution to the specified height and width and return the actual frame size set by the webcam.

        :param h: int. Desired height of the frames.
        :param w: int. Desired width of the frames.
        :return: tuple. The final frame size (height, width) after attempting to set the desired resolution.
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        return self._h, self._w

    def _is_resolution_natively_supported(self, h: int, w: int):
        current_h, current_w = self._h, self._w
        supported_h, supported_w = self._set_webcam_resolution(h=h, w=w)
        new_h, new_w = self._set_webcam_resolution(h=current_h, w=current_w)
        assert new_h == current_h and new_w == current_w, f"Webcam resolution could not be restored to {current_h}x{current_w}."
        return supported_h == h and supported_w == w

    def _max_available_resolution(self) -> Tuple[int, int]:
        """
        Get the maximum allowed webcam resolution without changing the current resolution.

        :return: tuple. The maximum allowed webcam resolution (height, width).
        """
        # Store the current resolution
        current_h, current_w = self._h, self._w

        # Set the webcam resolution to an extremely high value and get the resulting maximum allowed resolution
        max_h, max_w = self._set_webcam_resolution(h=1e6, w=1e6)

        # Restore the original resolution
        self._set_webcam_resolution(h=current_h, w=current_w)

        return max_h, max_w

    def _adjust_image_shape(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Adjust the shape of the frame according to the on_aspect_ratio_lost parameter.

        :param frame: np.ndarray. The input frame.
        :param h: int. Desired height of the frame.
        :param w: int. Desired width of the frame.
        :return: np.ndarray. The adjusted frame.
        """
        if self.on_aspect_ratio_lost == RESIZE:
            return cv2.resize(src=frame, dsize=(w, h))
        elif self.on_aspect_ratio_lost == CROP:
            return self._resize_and_center_crop(frame=frame, h=h, w=w)
        else:
            raise ValueError(f"Invalid value for 'on_aspect_ratio_lost' parameter: {self.on_aspect_ratio_lost}. "
                                f"Valid values are: {RESIZE}, {CROP}.")

    def _resize_and_center_crop(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Resize and center crop the input frame to the desired dimensions, while preserving the original aspect ratio.

        :param frame: np.ndarray. The input frame to be resized and cropped.
        :param h: int. The desired height of the output frame.
        :param w: int. The desired width of the output frame.
        :return: np.ndarray. The resized and center-cropped frame.
        """
        current_h, current_w = frame.shape[:2]
        aspect_ratio = current_w / current_h

        # Calculate the new dimensions such that the aspect ratio is preserved
        if float(h) / w > aspect_ratio:
            new_h, new_w = h, int(h * aspect_ratio)
        else:
            new_w, new_h = w, int(w / aspect_ratio)

        # Resize the frame to the new dimensions
        frame = cv2.resize(src=frame, dsize=(new_w, new_h))

        # Calculate the position of the center crop
        y1, x1 = (new_h - h) // 2, (new_w - w) // 2
        y2, x2 = y1 + h, x1 + w

        # Crop the frame
        return frame[y1:y2, x1:x2]

    def read(self) -> tuple[bool, np.ndarray|None]:
        """
        Read a frame from the webcam.

        If the webcam's returned frame size is different from the user-set size, the frame is automatically resized.

        :return: tuple. A boolean indicating whether the frame was read successfully, and the frame itself.
        """
        ret, frame = self.cap.read()

        if ret:
            # Extract the height and width of the frame
            h, w = frame.shape[:2]

            # Resize the frame if the webcam's returned frame size is different from the user-set size
            if (h, w) != (self.h, self.w):
                frame = self._adjust_image_shape(frame=frame, h=self.h, w=self.w)

            # Convert the frame from BGR to RGB format if necessary
            if not self.as_bgr:
                cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB, dst=frame)

        return ret, frame

    def stop(self):
        self.cap.stop()

    def get(self, propId):
        return self.cap.get(propId=propId)

    def set(self, propId, value):
        self.cap.set(propId=propId, value=value)

    def release(self):
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    # --------------- AUXILIARY METHODS ---------------
    def calculate_frame_size_keeping_aspect_ratio(self, h:int | None, w:int|None) -> tuple[int, int]:
        """
        When only one of the frame size dimensions is given, the other one is calculated keeping the
        aspect ratio with the original video.

        :param h: int or None. Height of the frame. If None, _w must be provided.
        :param w: int or None. Width of the frame. If None, _h must be provided.
        :return: The height and width of the frame.
        """


        if h is None and w is not None:
            h = int(round(self._h * w / self._w))
        elif h is not None and w is None:
            w = int(round(self._w * h / self._h))
        else:
            raise ValueError(f'Only one of the frame size dimensions must be provided.'
                             f' Got _h={h} and _w={w}.')
        assert h is not None and w is not None, f'Both _h and _w should have been calculated. Got _h={h} and _w={w}.'
        return h, w

    def __iter__(self):
        return self

    def __next__(self):
        return self.read_next_frame()

    def __del__(self):
        # Close the video
        if self._background:
            self.stop()
        self.release()
        self.cap = None
