import cv2
import numpy as np


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
#thresholdName = "Threshold.tif"
#imageName = "HumanMicrovascularEndothelialCellAdjusted.png"

#fibreBundle = cv2.imread("lightfield.tif") #Read lightfield image
#img = cv2.imread(imageName) #Read lightfield image


#width = fibreBundle.shape[1]
#height = fibreBundle.shape[0]
#dim = (width, height)
#img = cv2.resize(img, dim)
#cv2.imshow("image", img)
#createthreshold(fibreBundle)

#threshold = cv2.imread(thresholdName) #Need to read image in for multiple layers
#cv2.imshow("Threshold",threshold)
#img = cv2.imread(imageName) #Read in image to be overlayed
#shiftedImage = cv2.imread(imageName) #Read in image to be shifted

#height, width, channels = img.shape #get dimentions of image to be shifted

#-----------------Cardinal Wobble Image (Cannot be done all at the same time, Python related issues)-----------------#
#imgShiftDown = shiftImageDown(shiftedImage, img, 100)
#imgShiftUp = shiftImageUp(shiftedImage, img, 100)
#imgShiftRight = shiftImageRight(shiftedImage, img, 100)
#imgShiftLeft = shiftImageLeft(shiftedImage, img, 100)


#outputOriginal = overlapImages(threshold, img) #Overlap images to see image through cores
#cv2.imshow("Original Image", outputOriginal)

#outputShifted = overlapImages(threshold, imgShift) #Overlap images to see image through cores
#outputAverage = gaussian = cv2.GaussianBlur(outputShifted, (1,1), 8)
#cv2.imshow("Average Output", outputAverage)


#outputShifted = cv2.imread("ImgShiftedLeft 100.png")
#cv2.imshow("Shifted Output",outputShifted)
#cv2.imshow("Original", outputOriginal) #Display image through cores

#correctedShift = shiftImageRight(img, outputShifted, 100)
#cv2.imshow("Corrected Output Image", correctedShift)


#-----------------Image overlap-----------------#
OriginalImage = cv2.imread("ImgOriginalImage.png")

correctedUp = cv2.imread("ImgCorrectedUpShift.png")
correctedDown = cv2.imread("ImgCorrectedDownShift.png")
correctedLeft = cv2.imread("ImgCorrectedLeftShift.png")
correctedRight = cv2.imread("ImgCorrectedRightShift.png")

firstOrderCorrection = (OriginalImage/255 + correctedUp/255) *0.7
secondOrderCorrection = (OriginalImage/255 + correctedUp/255 + correctedDown/255) * 0.6
thirdOrderCorrection = (OriginalImage/255 + correctedUp/255 + correctedDown/255 + correctedLeft/255) * 0.5
fourthOrderCorrection = (OriginalImage /255 + correctedUp/255 + correctedDown/255 + correctedLeft/255 + correctedRight/255) *0.4

cv2.imshow("First Correction", firstOrderCorrection)
cv2.imshow("Second Correction", secondOrderCorrection)
cv2.imshow("Third Correction", thirdOrderCorrection)
cv2.imshow("Final Correction", fourthOrderCorrection)
cv2.imshow("Original Image", OriginalImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
