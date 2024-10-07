import cv2
import numpy as np
from PlateDetection import *

def segment_plate(cropped_region):       
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to improve segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Contour detection
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size (to isolate characters)
    character_contours = []
    for contour1 in contours:
        (x, y, w, h) = cv2.boundingRect(contour1)
        aspect_ratio = w / float(h)

        # Filter based on size and aspect ratio expected for characters, removing non-wanted blobs
        if 0.2 < aspect_ratio < 0.75 and h > 50:
            character_contours.append((x, y, w, h, contour1))

    # Sort the character contours by x-coordinate (ascending)
    character_contours = sorted(character_contours, key=lambda c: c[0])

    # Draw the contours of the sorted characters
    image_with_contours = cropped_region.copy()
    for _, _, _, _, contour in character_contours:
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

    # Extract and sort each character's image based on x-coordinate so that when recognizing them we get the correct order of elements
    character_images = []
    for x, y, w, h, _ in character_contours:
        character_image = binary_image[y:y + h, x:x + w]
        character_images.append(character_image)

    return character_images
