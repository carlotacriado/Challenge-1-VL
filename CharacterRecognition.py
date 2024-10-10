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

                # Find contours to crop so that the size of the template is just the shape of the bounding box around it.
                # This helps improve the results afterwards when applying template matching to the licence plates.
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


def recognize_plate(character_images, templates, threshold=0.7, scales=[0.8, 1.0, 1.2], angles=[-7, 0, 7]):
    # List to store (index, character, score) tuples
    matches = []  

    shear_range = 0.2 

    for idx, char_image in enumerate(character_images):
        best_match = None
        best_score = -1 
        
        # Iterate through all templates
        for char, template in templates.items():
            # Iterate over different scales for stretching/squeezing (vertically and horizontally)
            for scale_x in scales:
                for scale_y in scales:
                    scaled_template = cv2.resize(template, 
                                                 (int(template.shape[1] * scale_x), int(template.shape[0] * scale_y)))

                    # Resize the character image to match the scaled template size (This yields much better results)
                    resized_char_image = cv2.resize(char_image, 
                                                    (scaled_template.shape[1], scaled_template.shape[0]))

                    # Apply different rotations to the template
                    for angle in angles:
                        center = (scaled_template.shape[1] // 2, scaled_template.shape[0] // 2)
                        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated_template = cv2.warpAffine(scaled_template, rot_matrix, 
                                                          (scaled_template.shape[1], scaled_template.shape[0]))

                        for shear in [-shear_range, 0, shear_range]:
                            shear_matrix = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
                            affine_template = cv2.warpAffine(rotated_template, shear_matrix, 
                                                             (rotated_template.shape[1], rotated_template.shape[0]))

                            result = cv2.matchTemplate(resized_char_image, affine_template, cv2.TM_CCOEFF_NORMED)
                            _, score, _, _ = cv2.minMaxLoc(result)
                            
                            if score > best_score:
                                best_score = score
                                best_match = char

        if best_score > threshold:
            matches.append((idx, best_match, best_score))

    # Sort the matches by score in descending order and take the top 7
    top_matches = sorted(matches, key=lambda x: x[2], reverse=True)[:7]

    # Sort the top matches by their original index to retain the original order
    top_matches_sorted = sorted(top_matches, key=lambda x: x[0])

    # Concatenate the characters while preserving their order from the original images
    plate = ''.join([char for _, char, _ in top_matches_sorted])

    return plate
