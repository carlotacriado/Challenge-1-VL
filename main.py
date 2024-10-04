import cv2
import numpy as np
from PlateDetection import *



images_frontal = ["BaseImages/Frontal/067KSH.jpg","BaseImages/Frontal/1062FNT.jpg","BaseImages/Frontal/1565HTS.jpg",
                  "BaseImages/Frontal/2153GYX.jpg","BaseImages/Frontal/2929KXJ.jpg","BaseImages/Frontal/3340JMF.jpg",
                  "BaseImages/Frontal/3587DCX.jpg","BaseImages/Frontal/4674FHC.jpg","BaseImages/Frontal/5275HGY.jpg",
                  "BaseImages/Frontal/5488LKV.jpg","BaseImages/Frontal/5796DKP.jpg","BaseImages/Frontal/7153JWD.jpg",
                  "BaseImages/Frontal/8727JTC.jpg","BaseImages/Frontal/9247CZG.jpg","BaseImages/Frontal/9892JFR.jpg"]

images_lateral = ["BaseImages/Lateral/0182GLK.jpg","BaseImages/Lateral/0907JRF.jpg","BaseImages/Lateral/1498JBZ.jpg",
                  "BaseImages/Lateral/1556GMZ.jpg","BaseImages/Lateral/2344KJP.jpg","BaseImages/Lateral/3044JMB.jpg",
                  "BaseImages/Lateral/3587DCX.jpg","BaseImages/Lateral/3660CRT.jpg","BaseImages/Lateral/4674FHC.jpg",
                  "BaseImages/Lateral/5275HGY.jpg","BaseImages/Lateral/5789JHB.jpg","BaseImages/Lateral/5796DKP.jpg",
                  "BaseImages/Lateral/6000GVT.jpg","BaseImages/Lateral/6401JBX.jpg","BaseImages/Lateral/6554BNX.jpg",
                  "BaseImages/Lateral/6929LKK.jpg","BaseImages/Lateral/8727JTC.jpg"]

all_images = images_frontal+images_lateral

for image in all_images:
    plate_detected, cropped_region = find_plate(image)
    cv2.imshow("Image with plate detected", plate_detected)
    cv2.imshow("Cropped plate", cropped_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows

