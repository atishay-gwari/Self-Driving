# Self Driving

In this project we have leveraged some Computer Vision tools and architectures to make what can be used in an autonomous driving vehicle.

We have, in this project utilized both Traditional and Deep Learning based Computer Vision for the following purposes:

1. Lane Detection
2. Object Detection

### The Output

Combined output of lane and object detection.

https://user-images.githubusercontent.com/83054615/229808794-ac6bdaed-73af-4a23-ad3b-7e4df460ee55.mp4





### Lane Detection

We tried both traditional CV & Deep Learning based approach for this for curved lane detection and we found the traditional CV to have a more lighter and optimized output compared to its deep learning counterpart.

We tried using Mask RCNN for the lane detection which uses instance segmentation to segment out the lanes.
The model works but gives a very delayed and high latency output which does not seem ideal.

Traditional CV seemed to do the job just fine for us and we used it for our final project.

Lane Detection Pipeline:
- Distortion Correction
- Perspective Warp
- Sobel Filtering
- Histogram Peak Detection
- Sliding Window Search

> Although the script for the Mask RCNN is also provided in the lane-detection directory of the repository.
> The weights for the same can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1iUb0URArB2C6-X3DOzv80k0jRYzgccWB)

### Object Detection

Object detection in our project is handled by the YOLO V4 Sate-Of-The-Art model.

YOLO seemed to be the perfect candidate among object detection models because of its speed, pre-trained data variety and accuracy among other things.

Architecture of the YOLO V4 model.

![AO](https://user-images.githubusercontent.com/67366599/229793388-869c44d4-aced-4ddd-8c6b-7cbcd547eb7a.png)

Written in C and CUDA, DarkNet backbone is used since its light in weight and blazing fast.

The model successfully detects and localizes cars, pedestrians, traffic signs, traffic signals, bus, trucks and many more vehicular objects.
