import cv2
import numpy as np
from matplotlib import pyplot as plt

file_path = '/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/binary_img/binary (5).png'
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

#Using cv2 functions
# kernel = np.ones((3, 3), np.uint8)

# eroded_image = cv2.erode(image, kernel, iterations=3)
# dilated_image = cv2.dilate(image, kernel, iterations=3)
# opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)
# closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
# titles = ['Original Image', 'Eroded Image', 'Dilated Image', 'Opened Image', 'Closed Image']
# images = np.hstack([image, eroded_image, dilated_image, opened_image, closed_image])

# cv2.imwrite('/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Code W3/Result/MorphologyUsingCv2Funcs.png',images)
# cv2.imshow('Result', images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Using custom functions
def Ero(image):
    rows, cols= image.shape
    image1 = image * 0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            image1[i,j] = image[i,j] and image[i-1,j] and image[i+1,j] and image[i,j-1] and image[i,j+1]
    return image1

def Erosion(image, iter: int):
    for x in range(1, iter):
        image = Ero(image)

    return image

def Dil(image):
    rows, cols= image.shape
    image1 = image * 0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            image1[i,j] = image[i,j] or image[i-1,j] or image[i+1,j] or image[i,j-1] or image[i,j+1]
    return image1

def Dilation(image, iter: int):
    for x in range(1, iter):
        image = Dil(image)

    return image

def Opening(image, iter: int):
    for x in range(1, iter):
        image = Ero(image)
    for x in range(1, iter):
        image = Dil(image)
    return image

def Closing(image, iter: int):
    for x in range(1, iter):
        image = Dil(image)
    for x in range(1, iter):
        image = Ero(image)
    return image

images = np.hstack([image ,Erosion(image,3), Dilation(image,3), Opening(image,3), Closing(image,3)])

cv2.imshow('Result', images)
cv2.waitKey(0)
cv2.destroyAllWindows()