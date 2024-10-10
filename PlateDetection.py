import cv2
import numpy as np
# desired shape: (2268, 4032, 3)

def contains_blue(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 170, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(mask)
    if blue_pixels > 350:
        return True
    else:
        print("Could not find blue region")
        return False

def find_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35,35), 5)

    # Apply blackhat operation to highlight to reveal dark regions (numbers) over light regions (the plate itself)
    filterSize = (41, 41)   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel, iterations=1)

    # Apply blackhat operation again, this time using a rectangular kernel to hopefully remark the plate and not the rest of elements
    filterSize = (200, 5)  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_BLACKHAT, kernel, iterations=1)

    # Binarize the image so that the plate (and if lucky nothing else) remains white while the rest of the image is black
    _, binary = cv2.threshold(blackhat, 90, 255, cv2.THRESH_BINARY)

    filterSize = (11,11)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations= 5) # Dilate to make sure the rectangle will cover the whole plate in the original image
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations= 5) # Close operation so that tiny gaps are closed and achieve a much better resut

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in sorted_contours:
        # Get the bounding box for the current contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extend the bounding box by a margin to increase the area checked for blue (this yields better results, proved experimentally)
        margin_x = 200
        margin_y = 10
        x_extended = max(0, x - margin_x)
        y_extended = max(0, y - margin_y)
        w_extended = min(image.shape[1], x + w + margin_x) - x_extended
        h_extended = min(image.shape[0], y + h + margin_y) - y_extended

        # Crop the extended region from the original image
        region = image[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended].copy()

        # Check if the extended region contains blue
        if contains_blue(region):
            # Draw a rectangle around the blob that contains blue
            cv2.rectangle(image, (x_extended, y_extended), (x_extended + w_extended, y_extended + h_extended), (0, 255, 0), 3)

            return image, region
    else:
        # If no contour containing blue is found, display a message
        print("Could not find a plate")
        return None, None
