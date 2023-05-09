# Webcam
<img alt="webcam" title="webcam" src="https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/logo.png" width="20%" align="left"> **Webcam** is a simple, yet powerful Python library that brings advanced webcam handling to your projects. It offers numerous advantages over alternatives like [cv2.VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), making it a _go-to choice_ for developers seeking to enhance their webcam integration and usage.

With **Webcam**, developers can easily access and manage video input from various sources like _webcams_, _IP cameras_, and _video files_. The library provides options to configure *frame sizes* and *cropping* or *resizing* behavior directly in the constructor, ensuring that all returned frames have the desired format. Additionally, **Webcam** provides a unique feature that allows users to simulate *webcam input* using _video files_, perfect for testing and development.

Designed for **simplicity** and **user-friendliness**, **Webcam** provides advanced features without compromising **ease of use**. Allowing you to focus on building your application without getting bogged down by the intricacies of webcam management.


## Advanced Functionality
**Webcam** offers an extensive range of features, elevating basic webcam handling to a level of advanced, flexible control:

- **Enhanced Input Sources**: **Webcam** seamlessly handles all input types that [OpenCV](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a949d90b766ba42a6a93fe23a67785951) accepts, but with added versatility. Including _webcams_, _IP cameras_ (using RTSP URLs), and _video files_.

- **Customizable Frame Configuration**: With **Webcam**, you can define the specific **frame size** and manage **aspect ratio** changes when the camera's native resolution differs. Keep the widest field of view by setting just one dimension, or set both and decide between **center cropping** or **resizing** (sacrificing aspect ratio).


- **OpenCV Compatibility**: **Webcam** supports *RGB* and *BGR* reading, ensuring compatibility with your existing [OpenCV](https://opencv.org/) pipelines.

- **Background Reading**: Enhances performance by reading frames in a background thread, with an option to limit the frame rate.

- **Webcam Input Emulation**: Uses **video files** to accurately **emulate webcam input**. The frame you access at any instant matches the one from a **real-time webcam stream** at the same time, a feature that's especially handy for testing and development scenarios.

- **Iterator and Batch Processing**: **Webcam** offers an iterator for simplified camera frame reading, with optional batch yielding for efficient data handling.

Balancing simplicity and advanced functionality, Webcam is an efficient, flexible tool for webcam integration.

