"""
_perspective_manager.py: Contains the _PerspectiveManager class, which saves the homography matrix and
manages all the perspective-correction related operations, such as warping, after-warping cropping and
boundaries calculation

Author: Eric Canas
Github: https://github.com/Eric-Canas
Email: eric@ericcanas.com
Date: 20-06-2023
"""
from __future__ import annotations
from functools import lru_cache
import cv2
import numpy as np

INPUT, OUTPUT = 'input', 'output'

class _PerspectiveManager:
    def __init__(self, homography_matrix: np.ndarray|list[list[float], ...], default_w: int, default_h: int,
                 crop_boundaries: bool = False,
                 boundaries_color: tuple[float|int, float|int, float|int] = (0., 0., 0.)):
        self.default_w, self.default_h = default_w, default_h

        homography_matrix = np.array(homography_matrix, dtype=np.float32)
        self.homography_matrix = self.__apply_non_negative_translation_to_homography_matrix(m=homography_matrix,
                                                                                             w=default_w, h=default_h)
        # Inverse homography matrix is used for knowing the original place of coordinates in the warped image space
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)

        self.crop_boundaries = crop_boundaries
        self.boundaries_color = boundaries_color

    @property
    def output_w(self) -> int:
        w, h = self.calculate_output_shape(w=self.default_w, h=self.default_h)
        return w

    @property
    def output_h(self) -> int:
        w, h = self.calculate_output_shape(w=self.default_w, h=self.default_h)
        return h

    def warp(self, image: np.ndarray) -> np.ndarray:
        """
        Warp the image to generate the required perspective correction
        :param image: np.ndarray. Image to warp
        :return: np.ndarray. The output warped image
        """
        h, w = image.shape[:2]
        w_after_warp, h_after_warp = self.calculate_output_shape(w=w, h=h, cropping_boundaries=False)

        warped_image = cv2.warpPerspective(
            src=image,
            M=self.homography_matrix,
            dsize=(w_after_warp, h_after_warp),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.boundaries_color
        )

        if self.crop_boundaries:
            x_min, y_min, x_max, y_max = self.__get_crop_bbox_xyxy(h=h, w=w)
            warped_image = warped_image[y_min:y_max, x_min:x_max]

        return warped_image

    # Cache this function as most of times it will be always called with the same arguments
    @lru_cache(maxsize=16)
    def calculate_output_shape(self, w: int, h: int, cropping_boundaries: bool | None = None) -> tuple[int, int]:
        """
        Gets the new width and height that an image with width w, and height w will have after warping.
        It will depend on the defined self.homography_matrix and the value of cropping_boundaries.
        :param w: int. The width of the input image
        :param h: int. The height of the input image
        :param cropping_boundaries: int. If crop boundaries to hide black borders or not. If None, the value of
                                    self.crop_boundaries will be used.
        :return: tuple[int, int]. width and height of the image that .warp() will produce for an input image with
                                  of width w and height h.
        """
        if cropping_boundaries is None:
            cropping_boundaries = self.crop_boundaries

        # Get the new position for the four image corners
        warped_corners = self.__output_corners(w=w, h=h)
        # Get the four x coords and the four y coords
        x_coords, y_coords =  warped_corners[:, 0], warped_corners[:, 1]
        # If cropping boundaries, let's delete from here the larger and shorter limits, to ensure no black borders
        if cropping_boundaries:
            # Now they will contain only two coords, as we will crop largest limits to shortest ones.
            x_coords, y_coords = np.sort(x_coords)[1:-1], np.sort(y_coords)[1:-1]
        # Get the output width and height
        w_after_warp = int(np.ceil(np.max(x_coords)) - np.floor(np.min(x_coords)))
        h_after_warp = int(np.ceil(np.max(y_coords)) - np.floor(np.min(y_coords)))
        return w_after_warp, h_after_warp


    # ------------------------------ CALCULATE MAGNIFICATION FACTOR ------------------------------

    @lru_cache(maxsize=64)
    def get_hw_magnification_at_point(self, x:int|float, y:int|float) -> tuple[float, float]:
        """
        Get the magnification factor for the given point. It is calculated by setting two vectors extending along
        x and y axes around the point (x, y), a horizontal and a vertical one, with length 2. The vectors are then
        transformed by the homography matrix and the magnification factor is calculated as the ratio between the
        length of the transformed vectors and the original ones.
        :param x: int or float. The x coordinate of the point.
        :param y: int or float. The y coordinate of the point.
        :return: tuple[float, float]. The magnification factor for the given point. In the form (h_magnification, w_magnification).
        """
        assert self.homography_matrix is not None, "Homography matrix must be set before calling this method."
        assert self.homography_matrix.shape == (3, 3), f"Homography matrix must be a 3x3 matrix. Got {'x'.join(self.homography_matrix.shape)}"

        # Define two vectors extending along x and y axes around the point (x, y) with length 2
        vec_x = np.array(((x-1., y, 1.), (x, y, 1.), (x+1., y, 1.)), dtype=np.float32).T
        vec_y = np.array(((x, y-1., 1.), (x, y, 1.), (x, y+1., 1.)), dtype=np.float32).T

        # Apply the homography matrix to the two vectors
        vec_x_transformed = np.dot(self.homography_matrix, vec_x)
        vec_y_transformed = np.dot(self.homography_matrix, vec_y)

        # Convert the transformed coordinates from homogeneous to Cartesian coordinates
        vec_x_transformed /= vec_x_transformed[2]
        vec_y_transformed /= vec_y_transformed[2]

        # Compute the lengths of the transformed vectors
        length_transformed_x = np.linalg.norm(vec_x_transformed[:, -1] - vec_x_transformed[:, 0])
        length_transformed_y = np.linalg.norm(vec_y_transformed[:, -1] - vec_y_transformed[:, 0])

        # The magnification in x and y directions is the ratio of transformed length to original length
        homography_magnification_w = length_transformed_x / (vec_x[0, -1] - vec_x[0, 0]) if (vec_x[0, -1] - vec_x[
            0, 0]) != 0 else 1.0
        homography_magnification_h = length_transformed_y / (vec_y[1, -1] - vec_y[1, 0]) if (vec_y[1, -1] - vec_y[
            1, 0]) != 0 else 1.0

        return homography_magnification_w, homography_magnification_h

    def get_hw_magnification_for_line(self, xyxy_line: np.ndarray | tuple[int | float, int | float, int | float, int | float],
                                      space: str = INPUT) \
            -> tuple[float, float]:
        """
        Get the magnification factor for the given line. It is calculated by transforming the two endpoints of the line
        and calculating the ratio between the length of the transformed line and the original one.
        :param xyxy_line: np.ndarray or tuple[int|float, int|float, int|float, int|float]. The line to which the magnification
        factor must be calculated. It can be either a tuple with the coordinates of the two endpoints of the line or a
        numpy array with shape (2, 2) containing the coordinates of the two endpoints of the line.
        :param space: str. The space in which the line is defined. It can be either 'input' or 'output'. Default is 'output'.
        :return: tuple[float, float]. The magnification factor for the given line. In the form (h_magnification, w_magnification).
        """
        assert self.homography_matrix is not None, "Homography matrix must be set before calling this method."
        assert self.homography_matrix.shape == (
        3, 3), f"Homography matrix must be a 3x3 matrix. Got {'x'.join(map(str, self.homography_matrix.shape))}"

        if isinstance(xyxy_line, (tuple, list)):
            assert len(xyxy_line) == 4, f"Expected tuple of length 4, but got length {len(xyxy_line)}."
            xyxy_line = np.array(xyxy_line, dtype=np.float32)
        assert isinstance(xyxy_line, np.ndarray), "Line must be either a tuple or a numpy array."
        xyxy_line = xyxy_line.reshape(2, 2)

        # If line is given in output space, convert it. NOTE: That's dangerous. If using as module on webcam, if
        # the line is received in the output space, will probably mean that the resize has not been reverted.
        if space == OUTPUT:
            xyxy_line = self.output_space_points_to_input_space(points_xy=xyxy_line)

        # Transform the line using homography matrix
        line_transformed = np.dot(self.homography_matrix, np.c_[xyxy_line, np.ones(2)].T)
        line_transformed /= line_transformed[2]

        # Calculate the lengths in x and y directions separately for the original and transformed lines
        #dx_original, dy_original = np.abs(xyxy_line[1] - xyxy_line[0])
        #dx_transformed, dy_transformed = np.abs(np.diff(line_transformed[:2], axis=1).squeeze())

        # Calculate the magnification in x and y directions separately
        #magnification_w = dx_transformed / dx_original if dx_original != 0 else 1.0
        #magnification_h = dy_transformed / dy_original if dy_original != 0 else 1.0

        # TODO: We are calculating the magnification for the whole line, because h,w magnification becomes buggy when
        #  lines are too horizontal or vertical. We should find a way to fix that.
        # Calculate the lengths for the original and transformed lines
        len_original = np.sqrt(np.sum((xyxy_line[1] - xyxy_line[0]) ** 2))
        len_transformed = np.sqrt(np.sum(np.diff(line_transformed[:2], axis=1).squeeze() ** 2))

        # Calculate the magnification
        magnification = len_transformed / len_original if len_original != 0 else 1.0

        return magnification, magnification


    def output_space_points_to_input_space(
        self, points_xy: np.ndarray | tuple[int | float, int | float, int | float, int | float]) -> np.ndarray:
        """
        Transform the given line from output space to input space.

        :param points_xy: np.ndarray or tuple[int|float, int|float, int|float, int|float]. The line to be transformed.
        It can be either a tuple with the coordinates of the two endpoints of the line or a numpy array with shape (2, 2)
        containing the coordinates of the two endpoints of the line.

        :return: np.ndarray. The transformed line in input space. It has shape (2, 2).
        """
        assert self.inverse_homography_matrix is not None, "Homography matrix must be set before calling this method."
        assert self.inverse_homography_matrix.shape == (3, 3), \
            f"Inverse Homograph matrix must be a 3x3 matrix. Got {'x'.join(map(str, self.homography_matrix.shape))}"

        if isinstance(points_xy, (tuple, list)):
            points_xy = np.array(points_xy, dtype=np.float32)
        assert isinstance(points_xy, np.ndarray), "Line must be either a tuple, a list or a numpy array."
        points_xy = points_xy.reshape(-1, 2)
        n_points, coords = points_xy.shape
        assert coords == 2, f"Expected 2 coordinates per point, but got {coords}."

        if self.crop_boundaries:
            x_min, y_min, x_max, y_max = self.__get_crop_bbox_xyxy(h=self.default_h, w=self.default_w)
            points_xy += (x_min, y_min)
        # Transform the line back to the input space using the inverse homography matrix
        points_transformed = np.dot(self.inverse_homography_matrix, np.c_[points_xy, np.ones(n_points)].T)
        points_transformed /= points_transformed[2]
        points_transformed = points_transformed[:2].T

        return points_transformed


    # ------------------------------- AUXILIARY METHODS -------------------------------

    def __apply_non_negative_translation_to_homography_matrix(self, m: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Apply a translation to the given homography matrix to ensure that the minimum x and y coordinates
        are at (0,0) after transformation.
        :param m: np.ndarray. The matrix to be translated.
        :return: np.ndarray. The translated matrix.
        """
        corners = self.__build_corners(w=w, h=h)
        # Get the new position for the four image corners
        warped_corners = cv2.perspectiveTransform(corners[None, ...], m)[0]

        # Get the minimum x and y coordinates
        x_min = np.min(warped_corners[:, 0])
        y_min = np.min(warped_corners[:, 1])

        # Create a translation matrix to move the minimum x and y coordinates to (0,0)
        translation_matrix = np.array(((1., 0., -x_min), (0., 1., -y_min), (0., 0., 1.)), dtype=np.float32)

        # Combine the original matrix with the translation matrix
        combined_matrix = np.dot(a=translation_matrix, b=m)

        return combined_matrix

    @lru_cache(maxsize=16)
    def __get_crop_bbox_xyxy(self, h: int | float, w: int | float,
                             return_as_int: bool = True) -> tuple[int|float, int|float, int|float, int|float]:
        """
        Get the bounding box of the warped image that excludes the black borders. The returned bounding box is in the
        format x_min, y_min, x_max, y_max.
        :param h: int|float. Height of the image.
        :param w: int|float. Width of the image.
        :param return_as_int: bool. Whether to return the bounding box as integers or floats. Default: True. Set to
        False if you want subpixel precision.
        :return: tuple[int, int, int, int]. The bounding box of the warped image that excludes the black borders.
        """
        warped_corners = self.__output_corners(w=w, h=h)
        # Keep only the second and third x coords, as they are the left and right limits
        x_min, x_max = np.sort(warped_corners[:, 0])[1:-1]
        # Keep only the second and third y coords, as they are the top and bottom limits
        y_min, y_max = np.sort(warped_corners[:, 1])[1:-1]
        if return_as_int:
            x_min, x_max = int(np.floor(x_min)), int(np.ceil(x_max))
            y_min, y_max = int(np.floor(y_min)), int(np.ceil(y_max))
        assert y_min >= 0 and x_min >= 0, f"Cropping coords can't be negatives. Got x={x_min}, y={y_max}"
        assert y_max > y_min and x_max > x_min, f"x_max and y_max must be larger than x_min and y_min." \
                                                f"Got x_min={x_min}, x_max={x_max}, y_min={y_min} and y_max={y_max}"
        return x_min, y_min, x_max, y_max

    @lru_cache(maxsize=16)
    def __output_corners(self, w: int, h: int) -> np.ndarray:
        """
        Get the corners of the output image after warping an image with width w and height h.
        :param w: int. The width of the input image
        :param h: int. The height of the input image
        :return: np.ndarray. The corners of the output image after warping an image with width w and height h.
        """
        # Calculate the positions for each one of the four corners determined by w and h
        corners = self.__build_corners(w=w, h=h)
        # Get the new position for the four image corners
        warped_corners = cv2.perspectiveTransform(corners[None, ...], self.homography_matrix)[0]
        assert np.isclose(np.min(warped_corners), 0., atol=1e-3), f"Minimum warped corner should be 0. Got {np.min(warped_corners)}"
        return np.clip(warped_corners, a_min=0., a_max=None)

    @lru_cache(maxsize=32)
    def __build_corners(self, w: float | int, h: float | int) -> np.ndarray:
        """
        Build the corners of the image as a numpy array of shape (4, 2). In the format x1y1, x2y2, x3y3, x4y4.
        :param w: int|float. Width of the image.
        :param h: int|float. Height of the image.
        :return: np.ndarray. The corners of the image, as float32. Shape (4, 2). Format x1y1, x2y2, x3y3, x4y4.
        """
        w, h = float(w), float(h)
        return np.array(((0., 0.), (w - 1., 0.), (w - 1., h - 1.), (0., h - 1.)), dtype=np.float32)

