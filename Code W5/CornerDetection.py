import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt

file_path = "/Users/lonersmac/Documents/Docs/113-1/DJJ_Project/Pic/Peppers.png"

def corner_detection(file_path, k=0.04, sigma=2, threshold=0.01):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    kernel_x = np.array([[-1, 0, 1]]) 
    kernel_y = kernel_x.T              

    Ix = convolve2d(gray, kernel_x, mode="same")
    Iy = convolve2d(gray, kernel_y, mode='same')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    A = cv2.GaussianBlur(Ixx, (21, 21), sigmaX=sigma)
    B = cv2.GaussianBlur(Iyy, (21, 21), sigmaX=sigma)
    C = cv2.GaussianBlur(Ixy, (21, 21), sigmaX=sigma)

    R = A*B - C**2 - k*((A+B)**2)
    
    thr = threshold * R.max()
    corners = np.zeros_like(gray)
    corners[R > thr] = 255

    image_corners = image.copy()
    image_corners[R> thr] = [0, 0, 255]

    cv2.imshow("Harris Corners Detection", image_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

corner_detection(file_path, k=0.04, sigma=2, threshold=0.01)