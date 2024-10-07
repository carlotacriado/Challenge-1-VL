import cv2
import numpy as np
import os
from Segmentation import *

def load_templates(template_folder):
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            char = filename.split('.')[0]
            template_path = os.path.join(template_folder, filename)
            template_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)  # Load with alpha

            if template_image is not None:
                # Separate color channels and the alpha channel
                if template_image.shape[2] == 4:  # Check if there is an alpha channel
                    # Split the channels
                    b_channel, g_channel, r_channel, alpha_channel = cv2.split(template_image)
                    # Create a mask where the alpha channel is greater than 0 (visible areas)
                    visible_mask = alpha_channel > 0

                    grayscale_template = cv2.cvtColor(template_image[:, :, :3], cv2.COLOR_BGR2GRAY)
                    grayscale_template = np.where(visible_mask, grayscale_template, 255)

                    _, binary_template = cv2.threshold(grayscale_template, 253, 255, cv2.THRESH_BINARY)
                    binary_template = cv2.bitwise_not(binary_template)

                else:
                    # If the image doesn't have an alpha channel, use the grayscale method directly (this is just in case but should not affect us with the images we have)
                    grayscale_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
                    _, binary_template = cv2.threshold(grayscale_template, 127, 255, cv2.THRESH_BINARY)
                    binary_template = cv2.bitwise_not(binary_template)

                # Find contours
                contours, _ = cv2.findContours(binary_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Get the bounding box of the largest contour (assuming it's the character)
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

                    # Crop the binary template to the bounding box
                    cropped_template = binary_template[y:y+h, x:x+w]

                    # Save the cropped and processed template
                    templates[char] = cropped_template
                else:
                    print(f"Warning: No contours found in template {template_path}.")
            else:
                print(f"Warning: Template {template_path} could not be loaded.")

    return templates




def recognize_plate(character_images, templates, threshold=0.7):
    plate = "" #String where the plate will be stored (hopefully T^T)
    
    for char_image in character_images:
        best_match = None
        best_score = -1  # Initialize with a low score
        
        # Iterate through all templates
        for char, template in templates.items():
            # Resize the character image to match the template size if necessary
            resized_char_image = cv2.resize(char_image, (template.shape[1], template.shape[0]))
            #cv2.imshow("char", resized_char_image)
            #cv2.imshow("template", template)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows

            # Perform template matching
            result = cv2.matchTemplate(resized_char_image, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            
            # Check if the score is higher than the current best
            if score > best_score:
                best_score = score
                best_match = char

        # Add the best match to the plate string if the score is above the threshold
        if best_score > threshold:
            plate += best_match

    return plate