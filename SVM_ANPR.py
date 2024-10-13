import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
import os

def write_text(target_image, text):
    position = (200, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 6  # Font size
    color = (0, 255, 0)  # Text color (B, G, R) in white
    thickness = 12  # Thickness of the text

    # Draw a rectangle under the letters to improve readibility
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    rect_top_left = (position[0], position[1] - text_size[1] - 30)  
    rect_bottom_right = (position[0] + text_size[0], position[1] + 30) 
    cv2.rectangle(target_image, rect_top_left, rect_bottom_right, (0, 0, 0), -1)  # Black rectangle

    cv2.putText(target_image, text, position, font, font_scale, color, thickness)
    return target_image

all_images = []
for filename in os.listdir("FinalDataset/Frontal"):
    all_images.append(os.path.join("FinalDataset/Frontal", filename))

for filename in os.listdir("FinalDataset/Lateral"):
    all_images.append(os.path.join("FinalDataset/Lateral", filename))

# ---------- MODEL TRAINING ----------
# Function for extracting HOG characteristics from an image
def extract_hog_features(image):
    features, _ = hog(image,
                    orientations=9,        # Number of referrals
                    pixels_per_cell=(8, 8), # Cell size (8x8 pixels)
                    cells_per_block=(2, 2), # 2x2 cells per block
                    block_norm='L2-Hys',    # Block standardisation
                    transform_sqrt=True,    # Apply the sqrt transformation
                    visualize=True,         # Generate the HOG image for display
                    feature_vector=True)     # Return characteristics as a vector

    return features

# Function for loading an image dataset from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Make sure the image are the same size
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(filename[0])  # Suppose the file name begins with the class (e.g. ‘A_01.png’).
    return images, labels

# Load images and labels
folder_path = r'C:\Users\oleks\Downloads\Number_plates'  # Remplace par le chemin vers ton dataset
images, labels = load_images_from_folder(folder_path)

print("Image loaded")

# Extract the HOG characteristics for each image
hog_features = [extract_hog_features(img) for img in images]

# Encoding labels (characters)
le = LabelEncoder()
y = le.fit_transform(labels)

# Convert characteristics into a NumPy table
X = np.array(hog_features)

# Instantiating an SVM classifier with a linear kernel
svm = SVC(kernel='linear', probability=True)

# Train the SVM model
svm.fit(X, y)

print("Training done.")


def augment_image(image):
    augmented_images = []
    
    # Image rotation (between -10 and +10 degrees)
    for angle in range(-10, 11, 5):
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    
    # Image translation
    for shift in range(-5, 6, 5):
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(shifted)
    
    return augmented_images

# Apply data augmentation to all images
X_augmented = []
y_augmented = []

for i, img in enumerate(images):
    augmented_imgs = augment_image(img)
    hog_augmented = [extract_hog_features(aug_img) for aug_img in augmented_imgs]
    X_augmented.extend(hog_augmented)
    y_augmented.extend([y[i]] * len(augmented_imgs))

# Merge augmented data with original data
X = np.vstack((X, np.array(X_augmented)))
y = np.hstack((y, np.array(y_augmented)))

# Re-training SVM with augmented data
svm.fit(X, y)

print("Training completed with enhanced data.")

# --------- END OF TRAINING MODEL ----------

def contains_blue(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105, 170, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(mask)
    return blue_pixels > 350  # Ajusta este umbral según sea necesario

score = 0
total_score = len(all_images)

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
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations= 5) # Dilate to make sure the rectangle will cover the whole
                                                                               # plate in the original image
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations= 5) # Close operation so that tiny gaps are closed and achieve
                                                                              # a much better resut

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
        w_extended -= 100

        # Crop the extended region from the original image
        region = img[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended]

        # Check if the extended region contains blue
        if contains_blue(region):
            # Draw a rectangle around the blob that contains blue
            cv2.rectangle(img, (x_extended, y_extended), (x_extended + w_extended, y_extended + h_extended), (0, 255, 0), 3)

            # Display the result and exit the loop after detecting the first valid blob
            cv2.imshow("Blob with Blue Detected", cv2.resize(img, (800, 600)))
            registered_plate = img[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended]
            #cv2.imshow("Plate detected", registered_plate)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #ll ----- Segmentation -----
            
            # Convert the image to greyscale
            gray = cv2.cvtColor(registered_plate, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Binary thresholding to obtain a black and white image
            # Use adaptive thresholding or Otsu to adapt to lighting variations
            _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            #binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

            # Apply morphological operations to improve segmentation
            # Creation of a core for the opening (small white areas will be removed)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

            # Contour detection
            contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter outlines by size (to isolate characters)
            character_contours = []
            for contour1 in contours:
                (x, y, w, h) = cv2.boundingRect(contour1)
                aspect_ratio = w / float(h)

                # Filtrer selon la taille et le ratio d'aspect attendus pour les caractères
                if 0.1 < aspect_ratio < 0.75 and h_extended*0.9 > h and h > h_extended*0.3:  # Approximate values for characters
                    character_contours.append(contour1)

            character_contours = sorted(character_contours, key=lambda b:cv2.boundingRect(b)[0])

            # Draw the outlines of the characters found
            image_with_contours = registered_plate.copy()
            cv2.drawContours(image_with_contours, character_contours, -1, (0, 255, 0), 2)

            # Extract each character found in separate images
            character_images = []
            for contour2 in character_contours:
                x, y, w, h = cv2.boundingRect(contour2)
                character_image = binary_image[y:y + h, x:x + w]
                character_images.append(character_image)

            # Display results
            #cv2.imshow('Contours des caractères', image_with_contours)
            #cv2.imshow('Image morph', morph_image)
            #cv2.imshow('Image binary', binary_image)

            plate_number = image.split('\\')[-1].split('.')[0]
            plate_predicted = ''

            # To display each segmented character
            for i, char_img in enumerate(character_images):
                char_img = cv2.bitwise_not(char_img)
                char_img = cv2.resize(char_img, (64, 64))
                
                test_features = extract_hog_features(char_img).reshape(1, -1)

                # Predict the class of the test image
                prediction = svm.predict(test_features)
                probas = svm.predict_proba(test_features)
                confidence = np.max(probas) * 100  # Highest probability of predicted class
                predicted_label = le.inverse_transform(prediction)
                
                plate_predicted += predicted_label[0]
                if len(plate_predicted) > 7:
                    plate_predicted = plate_predicted[1:]

            predicted_img = write_text(img, plate_predicted)
            cv2.imshow("Prediction", cv2.resize(predicted_img, (800,600)))
            if plate_number == plate_predicted:
                score +=1

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            break
    else:
        # If no contour containing blue is found, display a message
        cv2.imshow("No Blue Detected", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
print(f'Actuel score: {score}')
print(f'Total of image: {total_score}')
print(f'Accuracy: {score*100/total_score}%.')
# I think if we take the biggest blue blob should work for all but one, to fix it if we add that it has to contain blue in the
# original image we should be set  :D