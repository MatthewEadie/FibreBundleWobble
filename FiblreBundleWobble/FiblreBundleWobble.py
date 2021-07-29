import cv2
import numpy as np
import os
import sys

np.set_printoptions(threshold=sys.maxsize)


#Threshold image and binarize
def createthreshold(fibreBundle):
    fibre_gray = cv2.cvtColor(fibreBundle, cv2.COLOR_BGR2GRAY) #convert image to gray scale
    median = cv2.medianBlur(fibre_gray, 1)
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



#------RESIZE IMAFE FOR PIXEL RESOLUTION------#
#imagePath = "C:/Users/uodit/source/repos/snowomi/FibreBundleWobble/FiblreBundleWobble/80788/Original/"
#imageSavePath = "C:/Users/uodit/source/repos/snowomi/FibreBundleWobble/FiblreBundleWobble/80788/Resized/"

#imageNames = os.listdir(imagePath)

##images have resolution 0.251 u/pixel
##fibre has resolution 0.32 u/pixel
##Need to reduce ize of image by 0.784 to match pixel resolution
#for imageName in imageNames:
#    print(imageName)
#    image = cv2.imread(imagePath + imageName)
#    imageResized = cv2.resize(image,(1345,738));
#    cv2.imwrite(imageSavePath + imageName, imageResized)




















#-----------------Main program-----------------#
#thresholdName = "Threshold-1820x1337.tif"
#threshold = cv2.imread(thresholdName, cv2.IMREAD_GRAYSCALE)

#imageName = "Image"
#testImageName = "TestImage"


#----Resize image to same as fibre bundle----#
#for i in range(1,11):
#    img = cv2.imread(imageName + str(i) + ".tif") #Read lightfield image

#    width = fibreBundle.shape[1]
#    height = fibreBundle.shape[0]
#    dim = (width, height)
#    img = cv2.resize(img, dim)
#    cv2.imwrite("Output" + imageName + str(i) + ".tif", img)


#for i in range(1,4):
#    img = cv2.imread(testImageName + str(i) + ".tif") #Read lightfield image

#    width = fibreBundle.shape[1]
#    height = fibreBundle.shape[0]
#    dim = (width, height)
#    img = cv2.resize(img, dim)
#    cv2.imwrite("Output" + testImageName + str(i) + ".tif", img)



#----Overlay training images and test images with fibre----#
#imageName = "OutputImage"
#testImageName = "OutputTestImage"

#for i in range(1,11):
#    img = cv2.imread(imageName + str(i) + ".tif", cv2.IMREAD_GRAYSCALE) #Read lightfield image
#    outputOverlap = overlapImages(threshold, img) #Overlap images to see image through cores
#    cv2.imshow("Image"+str(i), outputOverlap)


#for i in range(1,4):
#    img = cv2.imread(testImageName + str(i) + ".tif", cv2.IMREAD_GRAYSCALE) #Read lightfield image
#    outputOverlap = overlapImages(threshold, img) #Overlap images to see image through cores
#    cv2.imshow("FibreTestImage"+str(i), outputOverlap)










#-----------------Cardinal Wobble Image (Cannot be done all at the same time, Python related issues)-----------------#
#shiftedImage = cv2.imread(imageName) #Read in image to be shifted
#height, width, channels = img.shape #get dimentions of image to be shifted

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
#OriginalImage = cv2.imread("ImgOriginalImage.png")

#correctedUp = cv2.imread("ImgCorrectedUpShift.png")
#correctedDown = cv2.imread("ImgCorrectedDownShift.png")
#correctedLeft = cv2.imread("ImgCorrectedLeftShift.png")
#correctedRight = cv2.imread("ImgCorrectedRightShift.png")

#firstOrderCorrection = (OriginalImage/255 + correctedUp/255) *0.7
#secondOrderCorrection = (OriginalImage/255 + correctedUp/255 + correctedDown/255) * 0.6
#thirdOrderCorrection = (OriginalImage/255 + correctedUp/255 + correctedDown/255 + correctedLeft/255) * 0.5
#fourthOrderCorrection = (OriginalImage /255 + correctedUp/255 + correctedDown/255 + correctedLeft/255 + correctedRight/255) *0.4

#cv2.imshow("First Correction", firstOrderCorrection)
#cv2.imshow("Second Correction", secondOrderCorrection)
#cv2.imshow("Third Correction", thirdOrderCorrection)
#cv2.imshow("Final Correction", fourthOrderCorrection)
#cv2.imshow("Original Image", OriginalImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
