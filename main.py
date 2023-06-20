import cv2
from webcam import Webcam
import os
import numpy as np

FRAMES_TO_READ = 500

HOMOGRAPHY_MATRIX = np.array([[1.1, 0.05, -50],
                              [-0.05, 1, 50],
                              [0, 0.0005, 1]], dtype=np.float32)

if __name__ == '__main__':

    # Instantiate a Webcam instance with the background parameter
    webcam = Webcam(src=os.path.join('resources', 'test_video.mp4'), w=640, run_in_background=True,
                    homography_matrix=HOMOGRAPHY_MATRIX, crop_on_warping=False)

    # Iteratively read FRAMES_TO_READ frames
    for i, frame in zip(range(FRAMES_TO_READ), webcam):
        # Show the frames in a cv2 window
        cv2.imshow('Webcam Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources and close the window
    webcam.release()
    cv2.destroyAllWindows()
