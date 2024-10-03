import cv2
import numpy as np
from Computervision import *

images_frontal = ["Base Images/Frontal/images/067KSH.jpg","Base Images/Frontal/images/1062FNT.jpg","Base Images/Frontal/images/1565HTS.jpg",
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

for image in all_images:
    plate_detected, region = find_plate(image)
    cv2.imshow("Image with plate detected", plate_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows