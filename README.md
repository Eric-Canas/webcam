# Webcam
<img alt="webcam" title="webcam" src="https://raw.githubusercontent.com/Eric-Canas/webcam/main/resources/logo.png" width="20%" align="left"> **Webcam** is a simple, yet powerful Python library that brings advanced webcam handling to your projects. It offers numerous advantages over alternatives like [cv2.VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), making it a _go-to choice_ for developers seeking to enhance their webcam integration and usage.

With **Webcam**, developers can easily access and manage video input from various sources like _webcams_, _IP cameras_, and _video files_. The library provides options to configure *frame sizes* and *cropping* or *resizing* behavior directly in the constructor, ensuring that all returned frames have the desired format. Additionally, **Webcam** provides a unique feature that allows users to simulate *webcam input* using _video files_, perfect for testing and development.

At its core, **Webcam** is designed to be intuitive and user-friendly while providing advanced functionality and flexibility, making it an essential library for any project requiring webcam integration. The library simplifies webcam-related tasks and ensures a smooth, hassle-free experience, allowing you to focus on building your application without getting bogged down by the intricacies of webcam management.

Some notable features provided by the library include:

- Support for _webcam_, _IP cameras_ (using RTSP URLs), and _video files_ as input sources.
- Ability to precisely configure frame size and behavior when camera's native resolution is not compatible.
- Support for reading frames in both *RGB* and *BGR* formats.
- Batch processing of frames, where the iterator can yield multiple frames at once.
- Reading frames in a background thread for improved performance.
- Limiting the frame rate to a specified maximum value.
