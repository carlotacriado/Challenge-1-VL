import cv2
import numpy as np
import matplotlib.pyplot as plt
from PlateDetection import *
from Segmentation import *
from CharacterRecognition import *
import os

images_frontal = ["BaseImages/Frontal/7067KSH.jpg", "BaseImages/Frontal/1062FNT.jpg", "BaseImages/Frontal/1565HTS.jpg",
                  "BaseImages/Frontal/2153GYX.jpg", "BaseImages/Frontal/2929KXJ.jpg", "BaseImages/Frontal/3340JMF.jpg",
                  "BaseImages/Frontal/3587DCX.jpg", "BaseImages/Frontal/4674FHC.jpg", "BaseImages/Frontal/5275HGY.jpg",
                  "BaseImages/Frontal/5488LKV.jpg", "BaseImages/Frontal/5796DKP.jpg", "BaseImages/Frontal/7153JWD.jpg",
                  "BaseImages/Frontal/8727JTC.jpg", "BaseImages/Frontal/9247CZG.jpg", "BaseImages/Frontal/9892JFR.jpg"]

images_lateral = ["BaseImages/Lateral/0182GLK.jpg", "BaseImages/Lateral/0907JRF.jpg", "BaseImages/Lateral/1498JBZ.jpg",
                  "BaseImages/Lateral/1556GMZ.jpg", "BaseImages/Lateral/2344KJP.jpg", "BaseImages/Lateral/3044JMB.jpg",
                  "BaseImages/Lateral/3587DCX.jpg", "BaseImages/Lateral/3660CRT.jpg", "BaseImages/Lateral/4674FHC.jpg",
                  "BaseImages/Lateral/5275HGY.jpg", "BaseImages/Lateral/5789JHB.jpg", "BaseImages/Lateral/5796DKP.jpg",
                  "BaseImages/Lateral/6000GVT.jpg", "BaseImages/Lateral/6401JBX.jpg", "BaseImages/Lateral/6554BNX.jpg",
                  "BaseImages/Lateral/6929LKK.jpg", "BaseImages/Lateral/8727JTC.jpg"]

asd = images_frontal+images_lateral



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

asd_sorted = sorted(asd, key=lambda x: os.path.basename(x))
all_images_sorted = sorted(all_images, key=lambda x: os.path.basename(x))

# Compare the sorted lists
if asd_sorted == all_images_sorted:
    print("The lists contain the same images.")
else:
    print("The lists are different.")


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
    print("Loading image from:", image_path)
    img = cv2.imread(image_path)    
    print(img.shape)

    plate_detected, cropped_region = find_plate(image_path)
    
    # Process and display the segmented elements
    elements_cropped = segment_plate(cropped_region)

    # Load the templates
    templates = load_templates(template_folder)

    # Recognize the plate from the segmented characters
    plate = recognize_plate(elements_cropped, templates, threshold=0.5)

    # Print the recognized plate
    #print("Recognized Plate:", plate)
    #print("Correct Plate:", correct_plate)

    if plate == correct_plate:
        correct_detection += 1

print(f"Correctly detected plates: {correct_detection} out of {len(all_plates_list)} \nAccuracy: {(correct_detection/len(all_plates_list))}")
    # The first 4 elements have to be numbers so keep that in mind --> TO DO!!
    # There can only be 7 elements, stop when limit is reached