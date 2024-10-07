import cv2
import numpy as np
import matplotlib.pyplot as plt
from PlateDetection import *
from Segmentation import *
from CharacterRecognition import *
import os

frontal_images = "BaseImages/Frontal"
lateral_images = "BaseImages/Lateral"

def load_images(path, list):
    for filename in os.listdir(path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(path, filename)
            list.append(img_path)
    return list

all_images = load_images(frontal_images, list=[])
all_images = load_images(lateral_images, all_images)

template_folder = "FontImages"

all_plates_list = []

for image in all_images:
    # Extract the plate name (e.g., "7067KSH" from "BaseImages/Frontal/7067KSH.jpg")
    plate_name = image.split('/')[-1].split('.')[0]
    
    # Add to dictionary
    all_plates_list.append(plate_name)

#def display_image(image, title="Image"):
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title(title)
    #plt.axis('off')  # Hide axis
    #plt.show()

correct_detection = 0

for image_path, correct_plate in zip(all_images, all_plates_list):
    img = cv2.imread(image_path)    
    plate_detected, cropped_region = find_plate(image_path)
    
    # Process and display the segmented elements
    elements_cropped = segment_plate(cropped_region)

    # Load the templates
    templates = load_templates(template_folder)

    # Recognize the plate from the segmented characters
    plate = recognize_plate(elements_cropped, templates, threshold=0.5)

    if plate == correct_plate:
        correct_detection += 1

print(f"Correctly detected plates: {correct_detection} out of {len(all_plates_list)} \nAccuracy: {(correct_detection/len(all_plates_list))}")