import numpy as np
import os

# Define paths
directory_loc = os.listdir("C://Users//Anirudh//Documents//GitHub//BoneFractureDetection//BoneFractureData2//bone fracture detection.v4-v4.yolov8//test//images")
image_path = 'C://Users//Anirudh//Documents//GitHub//BoneFractureDetection//BoneFractureData2//bone fracture detection.v4-v4.yolov8//test//images'
labels_path = 'C://Users//Anirudh//Documents//GitHub//BoneFractureDetection//BoneFractureData2//bone fracture detection.v4-v4.yolov8//test//labels'

# Check if paths exist
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The path {image_path} does not exist.")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"The path {labels_path} does not exist.")

# Load data
try:
    images = np.load()
    labels = np.load(labels_path)
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
