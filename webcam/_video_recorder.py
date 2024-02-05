import cv2
import os
from warnings import warn
from time import sleep, time
import numpy as np


class _VideoRecorder:
    def __init__(self, output_path: str, frame_size_hw: tuple[int, int],
                 not_override: bool = False,
                 fps: int = 30):
        """
        Initialize the _Recorder.

        :param output_path: str. Path to the output video file. If the file already exists, it will be overwritten.
        :param frame_size_hw: tuple[int, int]. Height and width of the input frames that will be saved.
        :param warn_if_file_exist: bool. If True, a warning will be printed if the output file already exists.
        :param fps: int. FPS of the output video.
        """
        if not output_path.endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + ".mp4"

        if os.path.isfile(output_path):
            if not not_override:
                os.remove(output_path)
                warn(f'Output file already exists. It will be overwritten: {output_path}.')
            else:
                num = 1
                new_output_path = output_path
                while os.path.isfile(new_output_path):
                    new_output_path = f"{os.path.splitext(output_path)[0]}_{num}.mp4"
                    num += 1
                output_path = new_output_path
                warn(f'Output file already exists. Saving to {output_path} instead.')

        assert fps > 0, f'fps must be greater than 0. Got {fps}.'
        if not isinstance(fps, int):
            warn(f'fps must be an integer. It will be converted to int: {fps} -> {int(fps)}.')
            fps = int(fps)

        self._recordin_hw = frame_size_hw
        self.video_writer = cv2.VideoWriter(filename=output_path,
                                            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                            frameSize=frame_size_hw[::-1],
                                            fps=fps)
        self.writing = False

    def write(self, frame: np.ndarray):
        """
        Write a frame to the output video.

        :param frame: np.ndarray. Frame to write.
        """
        h, w = frame.shape[:2]
        assert (h, w) == self._recordin_hw, f'Frame size must be {self._recordin_hw}. Got {frame.shape[:2]}.'
        self.video_writer.write(frame)

    def close(self):
        """
        Close the window.
        """
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            # Get time to wait for the last frame to be written.
            try:
                sleep(0.25)
            # Some (Windows) computers throw an OSError when sleeping.
            except OSError:
                warn('An OSError was raised when trying to sleep for 0.25 seconds. Doing active waiting instead.')
                start = time()
                while time() - start < 0.25:
                    pass
            self.video_writer.release()
        self.video_writer = None


    def __enter__(self):
        """
        Enter the with block.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the with block.

        Args:
            exc_type: Exception. The type of the exception.
            exc_val: Exception. The value of the exception.
            exc_tb: Exception. The traceback of the exception.
        """
        self.close()

    def __del__(self):
        """
        Destructor.
        """
        self.close()