# Traffic-Survalance-with-Computer-Vision-and-Deep-Learning
The system takes video footage of a highway as input and provides statistics like the count of vehicles and an average estimated speed of vehicles on the highway. 

The statistics provided by the system can have many applications. Like, pricing the billboards on a highway for advertisement, higher the count of vehicles, higher the price. Moreover, the government can use this statistic to know how many vehicles are entering a city each day. 

The system internally uses YOLO object detection algorithm for vehicle detection, followed by, Centroid Tracking algorithm for tracking the detected vehicles.

_____________________________________________________________________________________________________________________________

# Instructions for running the project
1. Download the model weights from: <a href="https://drive.google.com/open?id=135qLm2XX46M7zBpsUjlSKNNzKt-A-r4F">Model weights</a>
2. Place it in the directory "model_data"
3. Run main.py

Note: The size of the model is greater than 100mb, and Github doesn't permit me to upload files greater than 100mb. So, I have uploaded it on my drive and shared the link here.
_____________________________________________________________________________________________________________________________

# Packages used
1. Tensorflow
2. Keras
3. OpenCV
4. Numpy
