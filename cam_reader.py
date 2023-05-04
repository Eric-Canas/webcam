"""
WebcamReader is a wrapper for the cv2.VideoCapture class used to read frames from a webcam.

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 04-03-2023
"""

from __future__ import annotations
import cv2
import numpy as np
from warnings import warn
import time
from typing import Tuple

from vid2info.video.config import DEFAULT_VIDEO_FRAME_RATE
from vid2info.video.utils import calculate_frame_size_keeping_aspect_ratio
from vid2info.video._webcam_reader_multithread import _WebcamReaderMultithread


class WebcamReader:
    def __init__(
        self,
        webcam_src: int | str = 0,
        h: int | None = None,
        w: int | None = None,
        always_maximize_aspect_ratio: bool = False,
        as_bgr: bool = True,
        batch_size: int | None = None,
        run_in_background: bool = True,
        max_frame_rate: int | None = DEFAULT_VIDEO_FRAME_RATE,
    ):
        """
        Initialize the WebcamReader.

        :param webcam_src: int or str. The index of the webcam or its path.
        :param h: int or None. Desired height of the frames. If None and `w` is provided, the aspect ratio will be preserved.
        :param w: int or None. Desired width of the frames. If None and `h` is provided, the aspect ratio will be preserved.
        :param as_bgr: bool. If True, the frames will be returned in BGR format, otherwise in RGB format.
        :param batch_size: int or None. If not None, the iterator will yield batches of frames (B, H, W, C). If None, the iterator will yield single frames (H, W, C).
        :param run_in_background: bool. If True, the frames will be read in a background thread (speeding up the reading).
        :param max_frame_rate: int or None. The maximum frame rate to read the frames at. If None, the maximum frame rate will be used.
        """
        self._background = run_in_background
        self.cap = _WebcamReaderMultithread(src=webcam_src).start() if run_in_background else cv2.VideoCapture(webcam_src)
        self.as_bgr = as_bgr

        # Calculate and set output frame size
        self.frame_size_hw = self._calculate_and_set_desired_resolution(h, w)

        # Set batch size and frame rate attributes
        self.batch_size = batch_size
        self.start_timestamp = time.time()
        self.max_frame_rate = max_frame_rate
        self.last_frame_timestamp = self.start_timestamp

    @property
    def h(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def w(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

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

        return self.h, self.w

    def calculate_frame_size_keeping_aspect_ratio(self, h: int | None, w: int | None) -> Tuple[int, int]:
        """
        Calculate the output frame size based on the desired width and height while keeping the aspect ratio.

        :param h: int or None. Desired height of the frames.
        :param w: int or None. Desired width of the frames.
        :return: tuple. The final frame size (height, width) with the aspect ratio preserved.
        """
        max_h, max_w = self._max_available_resolution()

        if h is None and w is None:
            return max_h, max_w

        if h is None:
            aspect_ratio = max_h / max_w
            return int(w * aspect_ratio), w

        if w is None:
            aspect_ratio = max_w / max_h
            return h, int(h * aspect_ratio)

        return h, w

    def _is_resolution_natively_supported(self, h: int, w: int):
        current_h, current_w = self.h, self.w
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
        current_h, current_w = self.h, self.w

        # Set the webcam resolution to an extremely high value and get the resulting maximum allowed resolution
        max_h, max_w = self._set_webcam_resolution(h=1e6, w=1e6)

        # Restore the original resolution
        self._set_webcam_resolution(h=current_h, w=current_w)

        return max_h, max_w

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
            if (h, w) != self.frame_size_hw:
                frame = cv2.resize(src=frame, dsize=self.frame_size_hw[::-1])

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

        :param h: int or None. Height of the frame. If None, w must be provided.
        :param w: int or None. Width of the frame. If None, h must be provided.
        :return: The height and width of the frame.
        """
        return calculate_frame_size_keeping_aspect_ratio(original_h=self.h, original_w=self.w, new_h=h, new_w=w)

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
