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

from functools import lru_cache

import cv2
import numpy as np
import time

import os

from webcam._image_webcam import _ImageWebcam
from webcam._video_webcam import _VideoWebcam
from webcam._webcam_background import _WebcamBackground
from webcam._perspective_manager import _PerspectiveManager

CROP, RESIZE = 'crop', 'resize'
INPUT, OUTPUT = 'input', 'output'

class Webcam:
    def __init__(
        self,
        src: int | str = 0,
        h: int | None = None,
        w: int | None = None,
        as_bgr: bool = False,
        batch_size: int | None = None,
        run_in_background: bool = False,
        max_frame_rate: int | None = None,
        on_aspect_ratio_lost: str = CROP,
        homography_matrix: np.ndarray | list[list[float], ...] | None = None,
        crop_on_warping: bool = True,
        boundaries_color: tuple[int, int, int] | list[int, int, int] = (0, 0, 0),
        simulate_webcam: bool = True):
        """
        Initialize the WebcamReader.

        :param src: int or str. The index of the webcam or its path.
        :param h: int or None. Desired height of the frames. If None and `raw_w` is provided, the aspect ratio will be preserved.
        :param w: int or None. Desired width of the frames. If None and `raw_h` is provided, the aspect ratio will be preserved.
        :param as_bgr: bool. If True, the frames will be returned in BGR format, otherwise in RGB format.
        :param batch_size: int or None. If not None, the iterator will yield batches of frames (B, H, W, C). If None, the iterator will yield single frames (H, W, C).
        :param run_in_background: bool. If True, the frames will be read in a background thread (speeding up the reading).
        :param max_frame_rate: int or None. The maximum frame rate to read the frames at. If None, there will be no limitations on frame rating.
        :param on_aspect_ratio_lost: str. What to do if the aspect ratio of the frames is different from the desired aspect ratio. Valid values are: 'crop' and 'resize'.
        :param homography_matrix: np.ndarray or list[list[float], ...] or None. The homography matrix to warp the frames with.
        If passed, frames will be warped before any other processing.
        :param crop_on_warping: bool. Only applied when homography_matrix is given. Determines if there will be visible
        black perspective boundaries, or if the image will be cropped to hide them. Default: True
        :param boundaries_color: tuple[int, int, int] or list[int, int, int]. Only applied when artificial boundaries
        must appear in the image, (as for example, when homography_matrix is given and crop_on_warping is False).
        :param simulate_webcam: bool. Only applied on videos. If True, the video will be readed simulating a webcam source,
        (the frame you'll read won't be the next one, but the one you would expect from a real-time video streaming. If False,
        the video will be readed sequentially, frame by frame.
        """
        assert on_aspect_ratio_lost in (CROP, RESIZE), f"Invalid value for `on_aspect_ratio_lost`: {on_aspect_ratio_lost}." \
                                                       f" Valid values are: {CROP}, {RESIZE}"
        self._background = run_in_background
        self.on_aspect_ratio_lost = on_aspect_ratio_lost

        is_file = isinstance(src, str) and os.path.isfile(src)
        # Initialize it for videos if the source is a string and a file exists at the path
        if (is_file and _is_image_file(file_path=src)) or isinstance(src, np.ndarray):
            self.cap = _ImageWebcam(image_source=src)
        elif is_file and _is_video_file(file_path=src):
            self.cap = _VideoWebcam(video_path=src, simulate_webcam=simulate_webcam)
        # Otherwise assume it is a webcam (both webcam or RTSP stream)
        else:
            self.cap = cv2.VideoCapture(src) if not run_in_background else _WebcamBackground(src=src).start()
        self.as_bgr = as_bgr

        if homography_matrix is None:
            self.perspective_manager = None
        else:
            self.perspective_manager = _PerspectiveManager(homography_matrix=homography_matrix,
                                                           default_h=self.raw_h,
                                                           default_w=self.raw_w,
                                                           crop_boundaries=crop_on_warping,
                                                           boundaries_color=boundaries_color)

        # Calculate and set output frame size
        self.frame_size_hw = self._calculate_and_set_desired_resolution(h=h, w=w)

        # Set batch size and frame rate attributes
        self.batch_size = batch_size
        self.start_timestamp = time.time()
        self.max_frame_rate = max_frame_rate
        self.last_frame_timestamp = self.start_timestamp

    @property
    def raw_h(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def raw_w(self) -> int:
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

    def read(self, transform: bool = True) -> tuple[bool, np.ndarray|None]:
        """
        Read a frame from the webcam.

        If the webcam's returned frame size is different from the user-set size, the frame is automatically resized.

        :return: tuple. A boolean indicating whether the frame was read successfully, and the frame itself.
        """
        ret, frame = self.cap.read()

        if ret and transform:
            # Adjust the perspective (if needed). If homography matrix is not defined it will do nothing
            if self.perspective_manager is not None:
                frame = self.perspective_manager.warp(image=frame)
            # Get the height and width of the frame
            h, w = frame.shape[:2]
            # Resize the frame if the webcam's returned frame size is different from the user-set size
            if (h, w) != (self.h, self.w):
                frame = self.__adjust_image_shape(frame=frame, h=self.h, w=self.w)

        # Convert the frame from BGR to RGB format if necessary
        if ret and not self.as_bgr:
            cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB, dst=frame)

        return ret, frame

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
            time.sleep(max(0, (1/self.max_frame_rate)*0.9 - (time.time() - self.last_frame_timestamp)))
            self.last_frame_timestamp = time.time()

        return frames[0] if self.batch_size is None else np.stack(frames, axis=0)

    def __adjust_image_shape(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
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
            return self.__resize_and_center_crop(frame=frame, h=h, w=w)
        else:
            raise ValueError(f"Invalid value for 'on_aspect_ratio_lost' parameter: {self.on_aspect_ratio_lost}. "
                                f"Valid values are: {RESIZE}, {CROP}.")

    def __resize_and_center_crop(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Resize and center crop the input frame to the desired dimensions, while preserving the original aspect ratio.

        :param frame: np.ndarray. The input frame to be resized and cropped.
        :param h: int. The desired height of the output frame.
        :param w: int. The desired width of the output frame.
        :return: np.ndarray. The resized and center-cropped frame.
        """
        current_h, current_w = frame.shape[:2]
        aspect_ratio = current_w / current_h

        # Calculate the new dimensions such that the smaller dimension matches the desired size
        if current_h < current_w:
            new_h = h
            new_w = int(np.ceil(new_h * aspect_ratio))
        else:
            new_w = w
            new_h = int(np.ceil(new_w / aspect_ratio))

        # Resize the frame to the new dimensions
        frame = cv2.resize(src=frame, dsize=(new_w, new_h))

        # Calculate the position of the center crop
        y1, x1 = (new_h - h) // 2, (new_w - w) // 2
        y2, x2 = y1 + h, x1 + w

        # Crop the frame
        return frame[y1:y2, x1:x2]

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
    @property
    def __pre_resize_w(self) -> int:
        if self.perspective_manager is None:
            return self.raw_w
        return self.perspective_manager.output_w

    @property
    def __pre_resize_h(self) -> int:
        if self.perspective_manager is None:
            return self.raw_h
        return self.perspective_manager.output_h

    def _calculate_frame_size_keeping_aspect_ratio(self, h: int | None, w: int | None) -> tuple[int, int]:
        """
        When only one of the frame size dimensions is given, the other one is calculated keeping the
        aspect ratio with the original video.

        :param h: int or None. Height of the frame. If None, raw_w must be provided.
        :param w: int or None. Width of the frame. If None, raw_h must be provided.
        :return: The height and width of the frame.
        """

        if h is None and w is not None:
            h = int(round(self.__pre_resize_h * w / self.__pre_resize_w))
        elif h is not None and w is None:
            w = int(round(self.__pre_resize_w * h / self.__pre_resize_h))
        else:
            raise ValueError(f'Only one of the frame size dimensions must be provided.'
                             f' Got raw_h={h} and raw_w={w}.')
        assert h is not None and w is not None, f'Both raw_h and raw_w should have been calculated. Got raw_h={h} and raw_w={w}.'
        return h, w

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
            h, w = self._calculate_frame_size_keeping_aspect_ratio(h=h, w=w)

        # Change the resolution if supported (and keeps the maximum aspect ratio if only one dimension was provided)
        if self._is_resolution_natively_supported(h=h, w=w):
            self._set_webcam_resolution(h=h, w=w)

        return h, w

    def _set_webcam_resolution(self, h: int, w: int) -> tuple[int, int]:
        """
        Set the webcam resolution to the specified height and width and return the actual frame size set by the webcam.

        :param h: int. Desired height of the frames.
        :param w: int. Desired width of the frames.
        :return: tuple. The final frame size (height, width) after attempting to set the desired resolution.
        """

        # Set the webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        # Get the actual resolution set by the webcam
        return self.raw_h, self.raw_w

    def _is_resolution_natively_supported(self, h: int, w: int):
        """
        Check if the webcam supports the specified resolution natively.
        :param h: int. Desired height of the frames.
        :param w: int. Desired width of the frames.
        :return: bool. True if the webcam supports the specified resolution natively.
        """
        assert isinstance(h, int) and isinstance(w, int), f"Invalid resolution: {h}x{w}. Expected integers. " \
                                                          f"Got {type(h)}x{type(w)}."
        current_h, current_w = self.raw_h, self.raw_w
        supported_h, supported_w = self._set_webcam_resolution(h=h, w=w)
        new_h, new_w = self._set_webcam_resolution(h=current_h, w=current_w)
        assert new_h == current_h and new_w == current_w, f"Webcam resolution could not be restored to {current_h}x{current_w}."
        return supported_h == h and supported_w == w

    def _max_available_resolution(self) -> tuple[int, int]:
        """
        Get the maximum allowed webcam resolution without changing the current resolution.

        :return: tuple. The maximum allowed webcam resolution (height, width).
        """
        # Store the current resolution
        current_h, current_w = self.raw_h, self.raw_w

        # Set the webcam resolution to an extremely high value and get the resulting maximum allowed resolution
        max_h, max_w = self._set_webcam_resolution(h=1e6, w=1e6)

        # Restore the original resolution
        self._set_webcam_resolution(h=current_h, w=current_w)

        return max_h, max_w



    # ---------------------------- PIXEL MAGNIFICATION -----------------------------
    @property
    def pixel_magnification(self) -> float:
        return (self.pixel_magnification_h + self.pixel_magnification_w) / 2

    @property
    def pixel_magnification_h(self) -> float:
        m_h, m_w = self.get_magnification_hw(x=None, y=None)
        return m_h

    @property
    def pixel_magnification_w(self) -> float:
        m_h, m_w = self.get_magnification_hw(x=None, y=None)
        return m_w

    @lru_cache(maxsize=64)
    def get_magnification_hw(self, x: int | float | None = None, y: int | float | None = None) -> float:
        """
        Calculate the magnification of the pixel at (x, y). If x and y are not given, the magnification of the
        center pixel is calculated. (x, y) coordinates are only relevant if the perspective is adjusted.

        :param x: int or float or None. The x coordinate of the pixel.
        :param y: int or float or None. The y coordinate of the pixel.
        :return: float. The magnification of the pixel at (x, y).
        """

        # Calculate the homography magnification if appliable
        if self.perspective_manager is not None:
            x = self.__pre_resize_w // 2 if x is None else x
            y = self.__pre_resize_h // 2 if y is None else y
            magnification_h, magnification_w = self.perspective_manager.get_hw_magnification_at_point(x=x, y=y)
        # Or use the default magnification
        else:
            magnification_h, magnification_w = 1.0, 1.0

        input_h, input_w = self.__pre_resize_h, self.__pre_resize_w

        # Calculate the pixel magnification for each axis when adjusting size
        magnification_h, magnification_w = self.__calculate_resizing_magnification_hw(input_h=input_h, input_w=input_w,
                                                                                      pre_magnification_h=magnification_h,
                                                                                      pre_magnification_w=magnification_w)

        # Return the magnification of the pixel at (x, y)
        return magnification_h, magnification_w

    def get_line_hw_magnification(self, line_xyxy: np.ndarray | tuple[int|float, int|float, int|float, int|float],
                                  space: str = OUTPUT) \
            -> tuple[float, float]:
        """
        Get the magnification factor for the given line. It allows to take precise measurements on the image, without
        having to worry about the perspective distortion, aspect ratio modifications or resizing.
        :param line_xyxy: tuple. The line to measure, in the format (x1, y1, x2, y2).
        :param space: str. The space in which the line is defined. It can be either INPUT or OUTPUT.
        :return: tuple. The magnification factor for the line.
        """
        if space == OUTPUT:
            line_xyxy = self.output_space_points_to_input_space(points_xy=line_xyxy)

        # Calculate the homography magnification if appliable
        if self.perspective_manager is not None:
            magnification_h, magnification_w = self.perspective_manager.get_hw_magnification_for_line(xyxy_line=line_xyxy, space=INPUT)
        # Or use the default magnification
        else:
            magnification_h, magnification_w = 1.0, 1.0

        magnification_h, magnification_w = self.__calculate_resizing_magnification_hw(input_h=self.__pre_resize_h,
                                                                                      input_w=self.__pre_resize_w,
                                                                                      pre_magnification_h=magnification_h,
                                                                                      pre_magnification_w=magnification_w)

        return magnification_h, magnification_w

    @lru_cache(maxsize=64)
    def __calculate_resizing_magnification_hw(self, input_h: int, input_w: int, pre_magnification_h: float = 1.0,
                                              pre_magnification_w: float = 1.0) -> tuple[float, float]:
        """
        Calculate the pixel magnification for each axis when adjusting size
        :param input_h: int. The input height.
        :param input_w: int. The input width.
        :param pre_magnification_h: float. The previous magnification in height. Default is 1.0.
        :param pre_magnification_w: float. The previous magnification in width. Default is 1.0.
        :return: tuple[float, float]. The updated magnification in height and width.
        """
        if self.on_aspect_ratio_lost == RESIZE:
            magnification_h = pre_magnification_h * (self.h / input_h)
            magnification_w = pre_magnification_w * (self.w / input_w)

        elif self.on_aspect_ratio_lost == CROP:
            resize_ratio = self.h / input_h if input_h < input_w else self.w / input_w
            magnification_h = pre_magnification_h * resize_ratio
            magnification_w = pre_magnification_w * resize_ratio
        else:
            raise ValueError(f"Invalid value for 'on_aspect_ratio_lost' parameter: {self.on_aspect_ratio_lost}."
                             f" Valid values are '{RESIZE}' and '{CROP}'.")

        return magnification_h, magnification_w


    # -------------------------- REVERSE TRANSFORMATIONS --------------------------

    def output_space_points_to_input_space(self, points_xy: np.ndarray | tuple[float | int, ...] | list[float | int, ...]) ->\
            np.ndarray:
        """
        Transform the given points from output space to input space.

        :param points_xy: np.ndarray. Points to be transformed. It has shape (N, 2), where N is the number of points.
        :return: np.ndarray. The transformed points. It has shape (N, 2).
        """
        if isinstance(points_xy, (tuple, list)):
            points_xy = np.array(points_xy, dtype=np.float32)
        assert isinstance(points_xy, np.ndarray), "Line must be either a tuple or a numpy array."
        points_xy = points_xy.reshape(-1, 2)

        if self.on_aspect_ratio_lost == RESIZE:
            points_xy = self.__rollback_resize_for_points(points_xy=points_xy)
        elif self.on_aspect_ratio_lost == CROP:
            points_xy = self.__rollback_crop_for_points(points_xy=points_xy)
        else:
            raise ValueError(f"Invalid value for 'on_aspect_ratio_lost' parameter: {self.on_aspect_ratio_lost}. "
                             f"Valid values are: {RESIZE}, {CROP}.")

        # Undo the homography transformation using the Perspective Manager's method
        points_transformed = self.perspective_manager.output_space_points_to_input_space(points_xy=points_xy)

        return points_transformed

    def __rollback_resize_for_points(self, points_xy: np.ndarray) -> np.ndarray:
        """
        Rollback the resize transformation.

        :param points_xy: np.ndarray. Points to be transformed. It has shape (N, 2), where N is the number of points.
        :return: np.ndarray. The transformed points. It has shape (N, 2).
        """
        scale_x = self.perspective_manager.output_w / self.w
        scale_y = self.perspective_manager.output_h / self.h

        points_xy[:, 0] *= scale_x
        points_xy[:, 1] *= scale_y

        return points_xy

    def __rollback_crop_for_points(self, points_xy: np.ndarray) -> np.ndarray:
        """
        Rollback the crop transformation.

        :param points_xy: np.ndarray. Points to be transformed. It has shape (N, 2), where N is the number of points.
        :return: np.ndarray. The transformed points. It has shape (N, 2).
        """
        # The input dimensions that the crop received (the output of the warping
        input_h, input_w = self.perspective_manager.output_h, self.perspective_manager.output_w
        # The output dimensions that the crop should produce
        output_h, output_w = self.h, self.w

        aspect_ratio = input_w / input_h

        # Calculate the intermediate dimensions that the crop should produce (dimensions to resize before cropping)
        if input_h < input_w:
            new_h = output_h
            new_w = int(np.ceil(new_h * aspect_ratio))
        else:
            new_w = output_w
            new_h = int(np.ceil(new_w / aspect_ratio))

        # Calculate the scale factor of the resize
        scale_x = new_w / input_w
        scale_y = new_h / input_h

        # Copy the points to avoid changing the original array
        points_xy = points_xy.astype(np.float32)
        # Divide the points by the scale factor (to rollback the resize)
        points_xy[:, 0] /= scale_x
        points_xy[:, 1] /= scale_y

        # Calculate the position of the center crop
        y1, x1 = (new_h - output_h) // 2, (new_w - output_w) // 2

        # Translate the points
        points_xy[:, 0] += x1
        points_xy[:, 1] += y1

        return points_xy

# -------------------------- BUILT-IN METHODS --------------------------

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

def get_rtsp_url(ip: str, username: str, password: str, port: int = 554, channel: int = 1, stream: int = 0) -> str:
    """
    Generate an RTSP URL for connecting to an IP camera.

    :param ip: str. The IP address of the camera.
    :param username: str. The username for accessing the camera.
    :param password: str. The password for accessing the camera.
    :param port: int. The RTSP port (default: 554).
    :param channel: int. The camera channel (default: 1).
    :param stream: int. The stream type (default: 0).
    :return: str. The generated RTSP URL.
    """
    return f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={stream}"


def _is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image by its extension.

    :param file_path: str. Path to the file.
    :return: bool. True if the file is an image, False otherwise.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    _, extension = os.path.splitext(file_path)
    return extension.lower() in image_extensions

def _is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video by its extension.

    :param file_path: str. Path to the file.
    :return: bool. True if the file is a video, False otherwise.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".m4v", ".3gp"}
    _, extension = os.path.splitext(file_path)
    return extension.lower() in video_extensions