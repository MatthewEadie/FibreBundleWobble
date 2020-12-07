import cv2
import numpy as np

thresholdName = "Threshold.tif"

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


def shiftImageDown(shiftedImage, img, amount):
    row = 0
    for i in range(height):
        if i < amount:
            shiftedImage[i,:] *= 0
        else:
            shiftedImage[i,:] = img[row,:]
            row += 1

    return shiftedImage

def shiftImageUp(shiftedImage, img, amount):
    for i in range(height):
        if i < height - amount:
            shiftedImage[i,:] = img[i + amount,:]
        else:
            shiftedImage[i,:] *= 0

    return shiftedImage

def shiftImageRight(shiftedImage, img, amount):
    col = 0
    for i in range(width):
        if i < amount:
            shiftedImage[:,i] *= 0
        else:
            shiftedImage[:,i] = img[:,col]
            col += 1

    return shiftedImage

def shiftImageLeft(shiftedImage, img, amount):
    for i in range(width):
        if i < width - amount:
            shiftedImage[:,i] = img[:,i + amount]
        else:
            shiftedImage[:,i] *= 0

    return shiftedImage




#-----------------Main program-----------------#
fibreBundle = cv2.imread("lightfield.tif") #Read lightfield image

createthreshold(fibreBundle)

threshold = cv2.imread(thresholdName) #Need to read image in for multiple layers
img = cv2.imread("ColourImage.jpg") #Read in image to be overlayed
shiftedImage = cv2.imread("ColourImage.jpg") #Read in image to be shifted

height, width, channels = img.shape #get dimentions of image to be shifted

#-----------------Cardinal Wobble Image (Cannot be done all at the same time, Python related issues)-----------------#
#imgShift = shiftImageDown(shiftedImage, img, 100)
#imgShift = shiftImageUp(shiftedImage, img, 100)
#imgShift = shiftImageRight(shiftedImage, img, 100)
imgShift = shiftImageLeft(shiftedImage, img, 100)



#Should these be smoothed so that each core only has 1 colour in it?
#That way shifting and smoothing would change the core colour value
outputOriginal = overlapImages(threshold, img) #Overlap images to see image through cores
outputShifted = overlapImages(threshold, imgShift) #Overlap images to see image through cores



#cv2.imwrite("ShiftedDown.png", outputShifted) #Error cannot write image with type (double)

cv2.imshow("Original", outputOriginal) #Display image through cores
cv2.imshow("Shifted", outputShifted) #Display image through cores

cv2.waitKey(0)
cv2.destroyAllWindows()
