import cv2
import numpy as np


#Threshold image and binarize
def createthreshold(fibreBundle):
    _, threshold = cv2.threshold(fibreBundle, 46, 255, cv2.THRESH_BINARY)
    fibre_gray = cv2.cvtColor(fibreBundle, cv2.COLOR_BGR2GRAY) #convert image to gray scale
    #meanC = cv2.adaptiveThreshold(fibre_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 59, 3)
    gaussianC = cv2.adaptiveThreshold(fibre_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 1)

    #cv2.imshow("Threshold", threshold)
    #cv2.imshow("Mean C", meanC)
    #cv2.imshow("Gaussian C", gaussianC)
    cv2.imwrite("Threshold.tif", gaussianC)
    

#overlap images to get view through fibre
def overlapImages(threshold, img):
    output = img / threshold
    return output



#Main program
fibreBundle = cv2.imread("lightfield.tif") #Read lightfield image

createthreshold(fibreBundle) #Create binary threshold image of cores

threshold = cv2.imread("Threshold.tif") #Need to read image in for multiple layers
img = cv2.imread("ColourImage.jpg") #Read in image to be overlayed

output = overlapImages(threshold, img) #Overlap images to see image through cores

cv2.imshow("Output", output) #Display image through cores
cv2.waitKey(0)
cv2.destroyAllWindows()
