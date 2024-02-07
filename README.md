# Webcam
<img alt="webcam" title="webcam" src="https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/logo.png" width="20%" align="left"> **Webcam** is a simple, yet powerful Python library that brings advanced webcam handling to your projects. Implemented under the same interface than [cv2.VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), it heavily simplifies _high-level_ frame manipulation, providing an intuitive and versatile way to handle video input from a range of sources like _webcams_, _IP cameras_, and _video files_.

With **Webcam**, you can easily control the **frame size**, adjust the **aspect ratio**, and apply **Perspective Transforms** to the _video stream_, all while preserving crucial information about **pixel origin** and **magnification**.

Furthermore, it includes a unique aspect that allows users to authentically **mirror webcam input** using _video files_. It ensures that the frame accessed at any particular moment matches the exact frame you'd encounter in a **real-time webcam stream**, making it a valuable tool for testing and development.

Designed for **simplicity** and **user-friendliness**, **Webcam** provides advanced features without compromising **ease of use**. This allows you to channel your efforts into developing your application, instead of wrestling with webcam control and frame manipulation details..

## Advanced Functionality
**Webcam** showcases a rich set of features, advancing beyond fundamental webcam manipulation to offer a higher degree of control and flexibility:

- **Enhanced Input Sources**: **Webcam** seamlessly handles all input types that [OpenCV](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a949d90b766ba42a6a93fe23a67785951) accepts, but with added versatility. Including _webcams_, _IP cameras_ (using RTSP URLs), and _video files_.

- **Customizable Frame Configuration**: With **Webcam**, you can define the specific **frame size** and manage **aspect ratio** changes when the camera's native resolution differs. Keep the widest field of view by setting just one dimension, or set both and decide between **center cropping** or **resizing** (sacrificing aspect ratio).

- **Perspective Transform**: Initialize the **Webcam** object with a **Homography Matrix** to maintain consistent perspective correction in every frame, while keeping the specified _frame size_.  If the perspective is deformed, you have the choice to **crop** the frame, removing black borders, or display the entire trapezoid to avoid losing any information.

- **Reversibility**: **Webcam** can easily retrieve _original coordinates_ and _pixel magnification_ for any point or section of the image. This is a valuable feature for **Computer Vision Engineers** needing to perform calculations based on the **raw camera sensor data**.

- **Background Reading**: Enhances performance by reading frames in a background thread, with an option to limit the frame rate.

- **Webcam Input Emulation**: You can use **video files** to accurately **emulate webcam input**. The frame you access at any instant will match the one from a **real-time webcam stream** at the same time. Especially handy for testing and development scenarios.

- **Iterator and Batch Processing**: **Webcam** offers an iterator for simplified camera frame reading, with optional batch yielding for efficient data handling.

Balancing simplicity and advanced functionality, Webcam is an efficient, flexible tool for webcam integration.


## Installation

You can install **Webcam** with

```bash
pip install webcam
```

## Usage

Reading a video stream is as simple as iterating over the defined **webcam** object.

```python
from webcam import Webcam

# Define a simple webcam object that will get video stream from webcam (src=0),
#  with a frame width of 640 (auto setting heigth to keep original aspect ratio)
webcam = Webcam(src=0, w=640)
print(f"Frame size: {webcam.w} x {webcam.h}")


for frame in webcam:
    # Show the frames in a cv2 window
    cv2.imshow('Webcam Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
```bash
>>> Frame size: 360 x 640
```

### Fitting to a strict frame size

Sometimes, your _Apps_ or _Computer Vision Models_ might need specific frame sizes that don't align with your video source's resolution. **Webcam** frees the user of having to deal with these frame resize details, especially when there's a discrepancy in the aspect ratio between the input and output.

```python
import os
from webcam import Webcam

# Let's use a video as a source instead of webcam
video_source = os.path.join('resources', 'test_video.mp4')

# Set an expected width and height, defining that "if aspect ratio differs", center crop the image.
no_deformation = Webcam(src=video_source, w=640, h=640, on_aspect_ratio_lost='crop')

# Replicate the situation, but defining that, "if aspect ratio differs", resize it, accepting the produced deformation.
deformation = Webcam(src=video_source, w=640, h=640, on_aspect_ratio_lost='resize')

# ---------------------------------------------------------------------------------------------------

# Print the original video resolution (output resolution will be 640 x 640 as specified)
print(f"Original WxH: {no_deformation.raw_w} x {no_deformation.raw_h}\n")

# Print the magnification in both cases (Cropped one will have a higher magnification on the input's shortest axis)
print(f"Resize WxH Magnification: {deformation.pixel_magnification_w} x {deformation.pixel_magnification_h}")
print(f"Center Crop WxH Magnification: {no_deformation.pixel_magnification_w} x {no_deformation.pixel_magnification_h}")

# Read the frames just as you would do it on OpenCV
grabbed, non_deformed_frame = no_deformation.read()
grabbed, deformed_frame = deformation.read()
```

```bash
Original WxH: 1280 x 720

Resize WxH Magnification: 0.50 x 0.89
Center Crop WxH Magnification: 0.89 x 0.89
```

| Input Frame (1280x720) | Resized (640x640) | Center Crop (640x640) |
|:-------------------------:|:-------------------------:|:-------------------------:|
| ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/strict_frame_size/base_frame.png) |  ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/strict_frame_size/resized_frame.png) | ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/strict_frame_size/center_crop_frame.png) |

## Applying Perspective Corrections

You can setup **Homography Matrices** to automatically apply _perspective transforms_. Wether if you apply them or not, you can keep the track of the origin of every coordinate in the raw sensor image.

```python
from webcam import Webcam

# Build an homography matrix that modifies the image perspective
homography_matrix = [[1., 0.1, 0.],
                     [0.1, 1., 0.],
                     [0.0002, -0.0001, 1.]]

# Build a webcam object passing this homography matrix (let's set a fixed height and let it decide width).
full_perspective = Webcam(src=video_source, h=640, homography_matrix=homography_matrix, crop_on_warping=False)
no_borders = Webcam(src=video_source, h=640, homography_matrix=homography_matrix, crop_on_warping=True)

# ---------------------------------------------------------------------------------------------------

# Let's calculate the origin of some coordinates in the raw input resolution (Note it expects a batch).
full_pers_x, full_pers_y = full_perspective.output_space_points_to_input_space(points_xy=((150, 225),))[0]
no_borders_x, no_borders_y = no_borders.output_space_points_to_input_space(points_xy=((625, 200), ))[0]

# Let's calculate the precise pixel magnification of two coords in an image. As perspective deformations
#  does not produce homogeneus magnifications as would happen when only resizes and/or crops are applied
no_borders_h_left, no_borders_w_left = no_borders.get_magnification_hw(x=150, y=225)
no_borders_h_right, no_borders_w_right = no_borders.get_magnification_hw(x=600, y=225)

print(f"Maginfication WxH at (150, 225): {no_borders_w_left} x {no_borders_h_left}")
print(f"Maginfication WxH at (600, 225): {no_borders_w_right} x {no_borders_h_right}\n")

print(f"Full Perspective: (150, 225) came from ({full_pers_x}, {full_pers_y})")
print(f"Cropped Perspective: (625, 200) came from ({no_borders_x}, {no_borders_y})")
```

```bash
>>> Maginfication WxH at (150, 225): 1.064 x 0.999
>>> Maginfication WxH at (600, 225): 0.984 x 0.842

>>> Full Perspective: (150, 225) came from (156.63, 258.30)
>>> Cropped Perspective: (625, 200) came from (733.69, 255.35)
```

| Input Frame (1280x720) | Full Perspective (943x640) | Hidding Borders (980x640) |
|:-------------------------:|:-------------------------:|:-------------------------:|
| ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/perspective/base_frame.png) |  ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/perspective/full_perspective.png) | ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/perspective/no_borders_perspective.png) |

### Limiting maximum frame rate
Sometimes, you'll need to simulate a fixed frame rate (for example, when you want to write video files). For these cases, you can limit the maximum frame rate. It will transform the frame retrieval in a blockant operation, making sure that you don't surpass the specified frame rate.

```python
from webcam import Webcam

# Build a webcam object that will limit frame rate to a maximum of 12 fps (get it as BGR).
webcam_12_fps = Webcam(src=video_source, max_frame_rate=12, as_bgr=True, simulate_webcam=True)

# ---------------------------------------------------------------------------------------------------

# Write the video on a .mp4 file that requires a fixed pre-known frame rate
video_writer = cv2.VideoWriter(filename='test.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                               fps=12, frameSize=(webcam_12_fps.w, webcam_12_fps.h))

for frame in webcam_12_fps:
    video_writer.write(frame)
video_writer.release()
```

| 5 FPS (1280x720) | 12 FPS (1280x720) |
|:-------------------------:|:-------------------------:|
| ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/limited_frame_rate/5_fps.gif) | ![](https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/usage_examples/limited_frame_rate/12_fps.gif) |

