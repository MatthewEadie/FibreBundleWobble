import cv2
import numpy as np

thresholdName = "Threshold.tiff"

#Threshold image and binarize
def createthreshold(fibreBundle):
    fibre_gray = cv2.cvtColor(fibreBundle, cv2.COLOR_BGR2GRAY) #convert image to gray scale
    median = cv2.medianBlur(fibre_gray, 1)
    gaussian = cv2.GaussianBlur(median, (3,3), 5)
    _, threshold = cv2.threshold(fibre_gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(thresholdName, threshold)
    
    
#overlap images to get view through fibre
def overlapImages(threshold, img):
    output = img / threshold
    return output


def shiftImage(shiftedImage, img):
    row = 0
    for i in range(height):
        if i < 100:
            shiftedImage[i,:] *= 0
        else:
            shiftedImage[i,:] = img[row,:]
            row += 1

    cv2.imshow("Original Image", img)
    cv2.imshow("Shifted Image", shiftedImage)


#Main program
fibreBundle = cv2.imread("lightfield.tif") #Read lightfield image

#createthreshold(fibreBundle)

threshold = cv2.imread(thresholdName) #Need to read image in for multiple layers
img = cv2.imread("ColourImage.jpg") #Read in image to be overlayed

shiftedImage = cv2.imread("ColourImage.jpg") #Read in image to be shifted

height, width, channels = img.shape

shiftImage(shiftedImage, img)

#output = overlapImages(threshold, img) #Overlap images to see image through cores

#cv2.imshow("Output", output) #Display image through cores
cv2.waitKey(0)
cv2.destroyAllWindows()
