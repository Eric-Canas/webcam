import cv2
from webcam import Webcam
import os

FRAMES_TO_READ = 500

if __name__ == '__main__':

    # Instantiate a Webcam instance with the background parameter
    webcam = Webcam(webcam_src=os.path.join('resources', 'test_video.mp4'), run_in_background=False)

    # Iteratively read FRAMES_TO_READ frames
    for i, frame in zip(range(FRAMES_TO_READ), webcam):
        # Show the frames in a cv2 window
        cv2.imshow('Webcam Frame', frame)
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources and close the window
    webcam.release()
    cv2.destroyAllWindows()
