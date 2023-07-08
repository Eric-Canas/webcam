# Webcam
<img alt="webcam" title="webcam" src="https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/logo.png" width="20%" align="left"> **Webcam** is a simple, yet powerful Python library that brings advanced webcam handling to your projects. Implemented under the same interface than [cv2.VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), it heavily simplifies _high-level_ frame manipulation, providing an intuitive and versatile way to handle video input from a range of sources like _webcams_, _IP cameras_, and _video files_.

**Webcam** grants you the power to dictate the **exact _frame size_** you want to read, regulate **_aspect ratio_**, and apply **Perspective Transforms** directly on the video stream, while retaining valuable information about **pixel origin** and **magnification** (useful for calculating image characteristics that depend on the camera sensor).

Furthermore, it includes a unique aspect that allows users to authentically **mirror webcam input** using _video files_. It ensures that the frame accessed at any particular moment matches the exact frame you'd encounter in a **real-time webcam stream**, making it a valuable tool for testing and development.

Designed for **simplicity** and **user-friendliness**, **Webcam** provides advanced features without compromising **ease of use**. This allows you to channel your efforts into developing your application, instead of grappling with the details and complexities of webcam control and frame manipulation.

## Advanced Functionality
**Webcam** showcases a rich set of features, advancing beyond fundamental webcam manipulation to offer a higher degree of control and flexibility:

- **Enhanced Input Sources**: **Webcam** seamlessly handles all input types that [OpenCV](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a949d90b766ba42a6a93fe23a67785951) accepts, but with added versatility. Including _webcams_, _IP cameras_ (using RTSP URLs), and _video files_.

- **Customizable Frame Configuration**: With **Webcam**, you can define the specific **frame size** and manage **aspect ratio** changes when the camera's native resolution differs. Keep the widest field of view by setting just one dimension, or set both and decide between **center cropping** or **resizing** (sacrificing aspect ratio).

- **Perspective Transform**: Initialize the **Webcam** object with a **Homography Matrix** to maintain consistent perspective correction in every frame, without having to worry about unexpected shifts or missmatches on the expected _frame size_.  If the perspective is deformed, you have the choice to **crop** the frame, removing black borders, or display the entire trapezoid to avoid losing any information.

- **Reversibility**: **Webcam** can reverse all applied transformations, allowing you to easily retrieve _original coordinates_ and _pixel magnification_ for any point or section of the image. This is a valuable feature for **Computer Vision Engineers** needing to perform calculations based on the **raw camera sensor data**.
  
- **OpenCV Compatibility**: **Webcam** supports *RGB* and *BGR* reading, ensuring compatibility with your existing [OpenCV](https://opencv.org/) pipelines.

- **Background Reading**: Enhances performance by reading frames in a background thread, with an option to limit the frame rate.

- **Webcam Input Emulation**: Uses **video files** to accurately **emulate webcam input**. The frame you access at any instant matches the one from a **real-time webcam stream** at the same time, a feature that's especially handy for testing and development scenarios.

- **Iterator and Batch Processing**: **Webcam** offers an iterator for simplified camera frame reading, with optional batch yielding for efficient data handling.

Balancing simplicity and advanced functionality, Webcam is an efficient, flexible tool for webcam integration.


## Installation

You can install **Webcam** with

```bash
pip install webcam
```
