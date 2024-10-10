import cv2
import numpy as np
from PlateDetection import *
from Segmentation import *
from CharacterRecognition import *
import os

def load_images(path, list):
    for filename in os.listdir(path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(path, filename)
            list.append(img_path)
    return list

def write_text(target_image, text):
    position = (200, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 4  # Font size
    color = (0, 255, 0)  # Text color (B, G, R) in white
    thickness = 8  # Thickness of the text

    cv2.putText(target_image, text, position, font, font_scale, color, thickness)
    return target_image

def crop_top(image, desired_shape=(2268, 4032)):
    desired_height, desired_width = desired_shape
    
    # Get current dimensions of the image
    original_height, original_width = image.shape[:2]
    
    # Ensure the image is large enough to crop
    if original_height < desired_height or original_width < desired_width:
        raise ValueError("Image size is smaller than the desired size!")
    
    # Crop the top of the image
    cropped_image = image[:desired_height, :desired_width]
    
    return cropped_image

frontal_images = "BaseImages/Frontal"
lateral_images = "BaseImages/Lateral"

own_frontal_images = "FinalDataset/Frontal"
own_lateral_images = "FinalDataset/Lateral"

own_database = "OwnDatabase"
single_image = ["4354LPC.jpg"]

all_images = load_images(frontal_images, list=[])
all_images = load_images(lateral_images, all_images)
our_images = load_images(own_database, [])
own_all_images = load_images(own_frontal_images, list=[])
own_all_images = load_images(own_lateral_images, own_all_images)

template_folder = "FontImages"

all_plates_list = []

for image in own_all_images:
    # Extract the plate name (e.g., "7067KSH" from "BaseImages/Frontal/7067KSH.jpg")
    plate_name = image.split('/')[-1].split('.')[0]
    
    # Add to dictionary
    all_plates_list.append(plate_name)

correct_detection = 0

for image_path, correct_plate in zip(own_all_images, all_plates_list):
    target_height, target_width = (2268, 4032)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (target_width, target_height))   

    plate_detected, cropped_region = find_plate(img)

    if cropped_region is None: # So if it fails to detect the image it won't crash the program
        print(f"Failed to detect plate for {image_path}. Skipping...")
        continue

    # Process and display the segmented elements
    elements_cropped = segment_plate(cropped_region)

    # Load the templates
    templates = load_templates(template_folder)

    # Recognize the plate from the segmented characters
    plate = recognize_plate(elements_cropped, templates, threshold=0.5)

    if plate == correct_plate:
        correct_detection += 1
    
    result_image = write_text(plate_detected, plate)
    cv2.imshow("Final result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    print(f"Plate Detected:{plate}")
    

print(f"Correctly detected plates: {correct_detection} out of {len(all_plates_list)} \nAccuracy: {(correct_detection/len(all_plates_list))}")
