import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from statistics import covariance

file_path_1 = '/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/BABOON.BMP'
file_path_2 = '/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/peppers256.bmp'
file_path_3 = '/Users/lonersmac/Documents/Docs/113-1/DJJ Project/Pic/BABOON_BRIGHT.bmp'

image1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(file_path_2, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(file_path_3, cv2.IMREAD_GRAYSCALE)

def NRMSE(img1, img2):
    if img1.shape != img2.shape:
        return "Error! Not the same dimension"
    else:
        diff = img2.astype(np.float32) - img1.astype(np.float32)
        row, col = img1.shape

        error_sum = np.float32(0)
        mean_sum = np.float32(0)

        for i in range(0, row):
            for j in range(0, col):
                error_sum += (diff[i][j].astype(np.float32))**2
                mean_sum += (img1[i][j].astype(np.float32))**2

        return (error_sum/mean_sum)**0.5

def PSNR(img1, img2):
    if img1.shape != img2.shape:
        return "Error! Not the same dimension"
    else:
        x_max = np.float32(255)
        diff = img2.astype(np.float32) - img1.astype(np.float32)
        row, col = img1.shape
        error_sum = np.float32(0)

        for i in range(0, row):
            for j in range(0, col):
                error_sum += (diff[i][j].astype(np.float32))**2
        mse = error_sum /np.float32((row+1)*(col+1))
        psnr = (math.log10(x_max**2 / mse)) * 10.0
        
        return psnr

def SSIM(img1, img2, c1=0.01, c2=0.03):
    if img1.shape != img2.shape:
        return "Error! Not the same dimension"
    else:
        L = np.float32(255)
        row, col = img1.shape
        error_sum = np.float32(0)
        mean_x = np.mean(img1)
        mean_y = np.mean(img2)
        var_x = np.var(img1)
        var_y = np.var(img2)
        cov_xy = np.cov(img1.flatten(), img2.flatten())[0, 1]

        # print(cov_xy)
        ssim = (2*mean_x*mean_y+(c1*L)**2) * (2*cov_xy+(c2*L)**2) / (mean_x**2+mean_y**2+(c2*L)**2) / (var_x+var_y+(c2*L)**2)
        return ssim

print("Result of image 1&3:")
print("NRMSE:",NRMSE(image1, image3))
print("PSNR:",PSNR(image1, image3))
print("SSIM:",SSIM(image1, image3))
print("Result of image 1&2:")
print("NRMSE:",NRMSE(image1, image2))
print("PSNR:",PSNR(image1, image2))
print("SSIM:",SSIM(image1, image2))