import cv2
import numpy as np


images_frontal = ["Base Images/Frontal/images/067KSH.jpeg","Base Images/Frontal/images/1062FNT.jpg","Base Images/Frontal/images/1565HTS.jpg",
          "Base Images/Frontal/images/2153GYX.jpg", "Base Images/Frontal/images/2929KXJ.jpg","Base Images/Frontal/images/3340JMF.jpg",
          "Base Images/Frontal/images/3587DCX.jpg", "Base Images/Frontal/images/4674FHC.jpg", "Base Images/Frontal/images/5275HGY.jpg",
          "Base Images/Frontal/images/5488LKV.jpg", "Base Images/Frontal/images/5796DKP.jpg", "Base Images/Frontal/images/7153JWD.jpg",
          "Base Images/Frontal/images/8727JTC.jpg",  "Base Images/Frontal/images/9247CZG.jpg", "Base Images/Frontal/images/9892JFR.jpg"]

images_lateral = ["Base Images/Lateral/images/0182GLK.jpg","Base Images/Lateral/images/0907JRF.jpg","Base Images/Lateral/images/1498JBZ.jpg",
                  "Base Images/Lateral/images/1556GMZ.jpg","Base Images/Lateral/images/2344KJP.jpg","Base Images/Lateral/images/3044JMB.jpg",
                  "Base Images/Lateral/images/3587DCX.jpg","Base Images/Lateral/images/3660CRT.jpg","Base Images/Lateral/images/4674FHC.jpg",
                  "Base Images/Lateral/images/5275HGY.jpg","Base Images/Lateral/images/5789JHB.jpg","Base Images/Lateral/images/5796DKP.jpg",
                  "Base Images/Lateral/images/6000GVT.jpg","Base Images/Lateral/images/6401JBX.jpg","Base Images/Lateral/images/6554BNX.jpg",
                  "Base Images/Lateral/images/6929LKK.jpg","Base Images/Lateral/images/8727JTC.jpg"]

all_images = images_frontal+images_lateral

def contains_blue(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 170, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(mask)
    return blue_pixels > 350  # Ajusta este umbral según sea necesario

for image in all_images:

    # Load the image
    img = cv2.imread(image)

    # Convert to grayscale so that we can apply the following operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply blur to get rid of noise in the image, it allows for cleaner results
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

    # Apply several morphological operations:
    filterSize = (11,11)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations= 5) # Dilate to make sure the rectangle will cover the whole plate in the original image
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations= 5) # Close operation so that tiny gaps are closed and achieve a much better resut


    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Iterate through the sorted contours
    for contour in sorted_contours:
        # Get the bounding box for the current contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extend the bounding box by a margin to increase the area checked for blue
        margin_x = 200
        margin_y = 10
        x_extended = max(0, x - margin_x)
        y_extended = max(0, y - margin_y)
        w_extended = min(img.shape[1], x + w + margin_x) - x_extended
        h_extended = min(img.shape[0], y + h + margin_y) - y_extended

        # Crop the extended region from the original image
        region = img[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended]

        # Check if the extended region contains blue
        if contains_blue(region):
            # Draw a rectangle around the blob that contains blue
            cv2.rectangle(img, (x_extended, y_extended), (x_extended + w_extended, y_extended + h_extended), (0, 255, 0), 3)

            # Display the result and exit the loop after detecting the first valid blob
            cv2.imshow("Blob with Blue Detected", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    else:
        # If no contour containing blue is found, display a message
        cv2.imshow("No Blue Detected", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# I think if we take the biggest blue blob should work for all but one, to fix it if we add that it has to contain blue in the
# original image we should be set  :D