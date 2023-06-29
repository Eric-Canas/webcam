from __future__ import annotations
import cv2
from webcam import Webcam
import os
import numpy as np
from matplotlib import pyplot as plt

FRAMES_TO_READ = 500

HOMOGRAPHY_MATRIX = np.array([[1.1, 0.05, -50],
                              [-0.05, 1, 50],
                              [0, 0.0005, 1]], dtype=np.float32)

def test_point_transformation(webcam, point: tuple[int|float]):
    # Convert points to tuple of int for cv2.line
    point = tuple(map(int, point))

    # Read frames without transformation
    ret, frame_no_transform = webcam.read(transform=False)
    # Read frames with transformation
    ret2, frame_with_transform = webcam.read(transform=True)

    # Transform points from output space to input space
    input_space_points = webcam.output_space_points_to_input_space(points_xy=point)
    input_space_point = tuple(map(int, input_space_points[0]))

    # Draw an X on the frame without transformation
    cv2.line(frame_with_transform, (point[0]-10, point[1]-10), (point[0]+10, point[1]+10), (0, 255, 0), 2)
    cv2.line(frame_with_transform, (point[0]-10, point[1]+10), (point[0]+10, point[1]-10), (0, 255, 0), 2)


    # Draw an X on the frame with transformation
    cv2.line(frame_no_transform, (input_space_point[0]-10, input_space_point[1]-10), (input_space_point[0]+10, input_space_point[1]+10), (0, 255, 0), 2)
    cv2.line(frame_no_transform, (input_space_point[0]-10, input_space_point[1]+10), (input_space_point[0]+10, input_space_point[1]-10), (0, 255, 0), 2)

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show the frame without transformation on the first subplot
    ax[1].imshow(frame_no_transform)
    ax[1].set_title("Frame without transformation")

    # Show the frame with transformation on the second subplot
    ax[0].imshow(frame_with_transform)
    ax[0].set_title("Frame with transformation")

    # Show the figure
    plt.show()

if __name__ == '__main__':

    # Instantiate a Webcam instance with the background parameter
    webcam = Webcam(src=os.path.join('resources', 'test_video.mp4'), w=640, run_in_background=True,
                    homography_matrix=HOMOGRAPHY_MATRIX, crop_on_warping=False, on_aspect_ratio_lost='resize')

    # Iteratively read FRAMES_TO_READ frames
    for i, frame in zip(range(FRAMES_TO_READ), webcam):
        # Show the frames in a cv2 window
        #cv2.imshow('Webcam Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Break the loop if the user presses the 'q' key
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        test_point_transformation(webcam=webcam, point=(0, 29, 0, 29))


    # Release the resources and close the window
    webcam.release()
    cv2.destroyAllWindows()
