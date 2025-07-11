# Situation-Analysis
The project involves developing a machine learning model capable of interpreting actions in CCTV footage by assigning a unique identifier to each detected person and categorizing their actions over time.
Project Overview
The project involves developing a machine learning model capable of
interpreting actions in CCTV footage by assigning a unique identifier to
each detected person and categorizing their actions over time. Key
functionalities include:
Person Detection: Identifying individuals within the CCTV footage.
Action Recognition: Classifying each individual’s actions (such as
standing, sitting, or walking) and recording these actions in a structured
dataset.
Time-Based Analysis: Logging the duration of each identified action for
further analysis.
Dataset Generation: Saving structured data with information on person
ID, action, and timestamp for additional insights and future analysis.
This project leverages computer vision techniques, such as pose
estimation and gesture recognition, to provide a comprehensive and
automated solution for interpreting human activities in surveillance
footage.
2 Tools and Technologies
2.1 Tools
Visual Studio Code: Used as the primary development environment for
coding, debugging, and testing the program.
Anaconda: Utilized for managing Python packages and dependencies to
ensure a seamless setup for the project.
Git: Version control for tracking changes and managing project
iterations.
TensorFlow Lite: Applied for efficient model deployment in resourceconstrained environments, enhancing performance during real-time
analysis.
2.2 Libraries
OpenCV: Essential for video capture and image processing, enabling
frame-by-frame analysis of CCTV footage.
MediaPipe: Used for pose detection, allowing us to identify and track
body landmarks and calculate actions like walking, sitting, and standing.
NumPy: Provides array support and efficient numerical computations
required for processing and analyzing landmark data.
Pandas: Used for handling data and organizing detected actions into a
structured dataset.
TensorFlow: Utilized for integrating any deep learning models used in
gesture or action recognition (if applicable).
2.3 Programming Language
Python: The primary programming language for this project due to its
versatility in handling machine learning libraries, ease of use with
OpenCV, and strong support for data manipulation through libraries like
NumPy and Pandas.
