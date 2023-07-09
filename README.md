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
print(f"Frame size: {webcam.w} x {webcam.h})

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

# Print the original video resolution (output resolution will be 640 x 640 as specified)
print(f"Original WxH: {no_deformation.raw_w} x {no_deformation.raw_h}\n")

# Print the magnification in both cases (Cropped one will have a higher magnification on the input's largest axis)
print(f"Resize WxH Magnification: {deformation.pixel_magnification_w} x {deformation.pixel_magnification_h}")
print(f"Center Crop WxH Magnification: {no_deformation.pixel_magnification_w} x {no_deformation.pixel_magnification_h}")
```

```bash
Original WxH: 1280 x 720


```

